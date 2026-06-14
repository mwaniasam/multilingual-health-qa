"""
=============================================================================
OVERNIGHT EXPERIMENT: 3-Phase Training Pipeline
=============================================================================
Run on Kaggle with GPU T4 x2. Designed to run UNATTENDED for 6-8 hours.
Saves submissions at each phase so even partial runs produce results.

Cell 1 (run first):
    !pip install -q sentence-transformers faiss-cpu rouge-score tqdm

Cell 2: paste this entire script
=============================================================================
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU, no DataParallel

import numpy as np
import pandas as pd
import torch
import faiss
import json
import time
import gc
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path('/kaggle/input/datasets/samuelmwania1/multilingual-health-qa-data/')
if not DATA_DIR.exists():
    DATA_DIR = Path('data/raw/')
OUTPUT_DIR = Path('/kaggle/working/')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path('submissions/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================================
# LOAD DATA
# ============================================================
log("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df = pd.read_csv(DATA_DIR / 'Val.csv')
test_df = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')
combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])
log(f"Combined: {len(combined)} samples")

questions_raw = combined['input'].fillna('').tolist()
answers_raw = combined['output'].fillna('').tolist()
subsets_raw = combined['subset'].tolist()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)


def evaluate_on_val(retrieve_fn, label=""):
    """Evaluate a retrieval function on val set. Returns (rouge1, rougeL)."""
    r1_scores, rl_scores = [], []
    val_qs = val_df['input'].fillna('').tolist()
    for idx in tqdm(range(len(val_df)), desc=f"Val eval [{label}]"):
        q = str(val_df.iloc[idx]['input']).strip()
        ref = str(val_df.iloc[idx]['output']).strip()
        subset = val_df.iloc[idx]['subset']
        pred = retrieve_fn(q, subset)
        r = scorer.score(ref, pred)
        r1_scores.append(r['rouge1'].fmeasure)
        rl_scores.append(r['rougeL'].fmeasure)
    r1 = np.mean(r1_scores)
    rl = np.mean(rl_scores)
    log(f"[{label}] Val ROUGE-1: {r1:.4f} | ROUGE-L: {rl:.4f}")
    return r1, rl


def make_submission(retrieve_fn, filename, comment):
    """Generate and save a submission CSV."""
    rows = []
    test_qs = test_df['input'].fillna('').tolist()
    for idx in tqdm(range(len(test_df)), desc=f"Submission [{filename}]"):
        q = str(test_df.iloc[idx]['input']).strip()
        subset = test_df.iloc[idx]['subset']
        answer = retrieve_fn(q, subset)
        if not answer:
            answer = "No answer available."
        rows.append({
            'ID': test_df.iloc[idx]['ID'],
            'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
        })
    sub = pd.DataFrame(rows)
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    path = OUTPUT_DIR / filename
    sub.to_csv(path, index=False)
    log(f"✅ Saved: {path} | Comment: {comment}")
    return path


# ============================================================
# PHASE 1: ITERATIVE HARD NEGATIVE MINING + FINE-TUNING
# ============================================================
log("=" * 70)
log("PHASE 1: ITERATIVE HARD NEGATIVE MINING + FINE-TUNING")
log("=" * 70)

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# --- Round 1: Initial training with mined hard negatives ---
log("\n--- Round 1: Loading base E5 model ---")
model = SentenceTransformer('intfloat/multilingual-e5-base', device='cuda:0')

log("Encoding corpus for hard negative mining...")
corpus_encoded = model.encode(
    [f"query: {q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True,
)
corpus_encoded = corpus_encoded.astype(np.float32)

log("Building FAISS index...")
index = faiss.IndexFlatIP(corpus_encoded.shape[1])
index.add(corpus_encoded)

log("Mining hard negatives (finding similar questions with different answers)...")
train_examples_hn = []
for i in tqdm(range(len(combined)), desc="Mining HN"):
    q = questions_raw[i]
    a = answers_raw[i]
    if not q.strip() or not a.strip():
        continue

    q_emb = corpus_encoded[i:i+1]
    D, I = index.search(q_emb, 30)  # top-30 candidates

    # Find hard negatives: similar question but DIFFERENT answer
    hard_negs = []
    for j in range(30):
        idx = int(I[0][j])
        if idx == i:
            continue  # skip self
        cand_a = answers_raw[idx]
        # Hard negative: high similarity but different answer
        if cand_a.strip() != a.strip() and len(hard_negs) < 3:
            hard_negs.append(f"passage: {cand_a}")

    if hard_negs:
        # Format: [anchor, positive, neg1, neg2, ...]
        texts = [f"query: {q}", f"passage: {a}"] + hard_negs
        train_examples_hn.append(InputExample(texts=texts))
    else:
        train_examples_hn.append(InputExample(texts=[f"query: {q}", f"passage: {a}"]))

log(f"Training examples with hard negatives: {len(train_examples_hn)}")

# Train Round 1
log("Training Round 1: 5 epochs with hard negatives, batch_size=8...")
train_loss = losses.MultipleNegativesRankingLoss(model)
train_dataloader = DataLoader(train_examples_hn, shuffle=True, batch_size=8)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=200,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5-hn-round1'),
    use_amp=True,
)
log("✅ Round 1 training complete!")

# Clean up
del train_examples_hn, train_dataloader, train_loss, corpus_encoded, index
gc.collect()
torch.cuda.empty_cache()

# --- Evaluate Round 1 ---
log("\nEncoding corpus with Round 1 model...")
corpus_r1 = model.encode(
    [f"query: {q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True,
)
corpus_r1 = corpus_r1.astype(np.float32)
index_r1 = faiss.IndexFlatIP(corpus_r1.shape[1])
index_r1.add(corpus_r1)

def retrieve_r1(q_text, subset):
    q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True).astype(np.float32)
    D, I = index_r1.search(q_emb, 10)
    for j in range(10):
        if str(combined.iloc[I[0][j]]['input']).strip() != q_text.strip():
            return str(combined.iloc[I[0][j]]['output'])
    return str(combined.iloc[I[0][0]]['output'])

r1_score_round1, _ = evaluate_on_val(retrieve_r1, "Round 1 (HN)")

# Save Round 1 submission
make_submission(retrieve_r1, "exp_overnight_round1.csv",
    f"Round 1: E5-base + hard negative mining, 5 epochs. Val ROUGE-1={r1_score_round1:.4f}")


# --- Round 2: Re-mine with improved model, retrain ---
log("\n--- Round 2: Re-mining hard negatives with improved model ---")
train_examples_hn2 = []
for i in tqdm(range(len(combined)), desc="Mining HN Round 2"):
    q = questions_raw[i]
    a = answers_raw[i]
    if not q.strip() or not a.strip():
        continue

    q_emb = corpus_r1[i:i+1]
    D, I = index_r1.search(q_emb, 30)

    hard_negs = []
    for j in range(30):
        idx = int(I[0][j])
        if idx == i:
            continue
        cand_a = answers_raw[idx]
        if cand_a.strip() != a.strip() and len(hard_negs) < 5:
            hard_negs.append(f"passage: {cand_a}")

    if hard_negs:
        texts = [f"query: {q}", f"passage: {a}"] + hard_negs
        train_examples_hn2.append(InputExample(texts=texts))
    else:
        train_examples_hn2.append(InputExample(texts=[f"query: {q}", f"passage: {a}"]))

log(f"Round 2 training examples: {len(train_examples_hn2)}")

log("Training Round 2: 3 epochs with fresh hard negatives...")
train_loss2 = losses.MultipleNegativesRankingLoss(model)
train_dataloader2 = DataLoader(train_examples_hn2, shuffle=True, batch_size=8)

model.fit(
    train_objectives=[(train_dataloader2, train_loss2)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5-hn-round2'),
    use_amp=True,
)
log("✅ Round 2 training complete!")

# Clean up
del train_examples_hn2, train_dataloader2, train_loss2, corpus_r1, index_r1
gc.collect()
torch.cuda.empty_cache()

# --- Evaluate Round 2 ---
log("\nEncoding corpus with Round 2 model...")
corpus_r2 = model.encode(
    [f"query: {q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True,
)
corpus_r2 = corpus_r2.astype(np.float32)
index_r2 = faiss.IndexFlatIP(corpus_r2.shape[1])
index_r2.add(corpus_r2)

def retrieve_r2(q_text, subset):
    q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True).astype(np.float32)
    D, I = index_r2.search(q_emb, 10)
    for j in range(10):
        if str(combined.iloc[I[0][j]]['input']).strip() != q_text.strip():
            return str(combined.iloc[I[0][j]]['output'])
    return str(combined.iloc[I[0][0]]['output'])

r1_score_round2, _ = evaluate_on_val(retrieve_r2, "Round 2 (HN iter)")

make_submission(retrieve_r2, "exp_overnight_round2.csv",
    f"Round 2: Iterative hard negative mining (2 cycles), 8 total epochs. Val ROUGE-1={r1_score_round2:.4f}")

# Save the E5 model for Phase 3
model.save(str(OUTPUT_DIR / 'e5-hn-final'))
log("✅ Phase 1 complete! E5 model saved.")


# ============================================================
# PHASE 2: CROSS-ENCODER RERANKER
# ============================================================
log("\n" + "=" * 70)
log("PHASE 2: CROSS-ENCODER RERANKER TRAINING")
log("=" * 70)

from sentence_transformers import CrossEncoder

log("Preparing cross-encoder training data...")
# For each question, create:
#   (question, correct_answer) → 1.0
#   (question, wrong_answer) → 0.0
ce_train_pairs = []
ce_train_labels = []

# Sample to keep training manageable (~100K pairs)
np.random.seed(42)
sample_indices = np.random.choice(len(combined), min(20000, len(combined)), replace=False)

for i in tqdm(sample_indices, desc="CE data"):
    q = questions_raw[i]
    a = answers_raw[i]
    if not q.strip() or not a.strip():
        continue

    # Positive pair
    ce_train_pairs.append([q, a])
    ce_train_labels.append(1.0)

    # Hard negative: retrieve similar question's answer
    q_emb = corpus_r2[i:i+1]
    D, I = index_r2.search(q_emb, 10)
    for j in range(10):
        idx = int(I[0][j])
        if idx == i:
            continue
        cand_a = answers_raw[idx]
        if cand_a.strip() != a.strip():
            ce_train_pairs.append([q, cand_a])
            ce_train_labels.append(0.0)
            break

    # Random negative
    rand_idx = np.random.randint(len(combined))
    while rand_idx == i:
        rand_idx = np.random.randint(len(combined))
    ce_train_pairs.append([q, answers_raw[rand_idx]])
    ce_train_labels.append(0.0)

log(f"Cross-encoder training pairs: {len(ce_train_pairs)}")

# Convert to InputExample format for CrossEncoder
ce_train_examples = []
for pair, label in zip(ce_train_pairs, ce_train_labels):
    ce_train_examples.append(InputExample(texts=pair, label=label))

log("Loading cross-encoder base model...")
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1, device='cuda:0')

log("Training cross-encoder for 3 epochs...")
ce_model.fit(
    train_dataloader=DataLoader(ce_train_examples, shuffle=True, batch_size=32),
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'ce-reranker'),
)
log("✅ Cross-encoder training complete!")

del ce_train_examples

# Clean up
del ce_train_pairs, ce_train_labels
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# PHASE 3: COMBINED PIPELINE — RETRIEVE + RERANK
# ============================================================
log("\n" + "=" * 70)
log("PHASE 3: COMBINED PIPELINE (E5-ft retrieve + CE rerank)")
log("=" * 70)

def retrieve_and_rerank(q_text, subset, top_k_retrieve=20, top_k_final=1):
    """Retrieve top-K with E5-ft, rerank with cross-encoder."""
    # Step 1: Retrieve with fine-tuned E5
    q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True).astype(np.float32)
    D, I = index_r2.search(q_emb, top_k_retrieve)

    candidates = []
    for j in range(top_k_retrieve):
        idx = int(I[0][j])
        cq = str(combined.iloc[idx]['input']).strip()
        if cq == q_text.strip():
            continue
        candidates.append({
            'answer': str(combined.iloc[idx]['output']),
            'e5_score': float(D[0][j]),
        })

    if not candidates:
        return str(combined.iloc[I[0][0]]['output'])

    # Step 2: Rerank with cross-encoder
    ce_pairs = [[q_text, c['answer']] for c in candidates]
    ce_scores = ce_model.predict(ce_pairs, show_progress_bar=False)

    # Combine scores: weighted sum
    for i, c in enumerate(candidates):
        c['ce_score'] = float(ce_scores[i])
        c['combined'] = 0.4 * c['e5_score'] + 0.6 * c['ce_score']

    candidates.sort(key=lambda x: -x['combined'])
    return candidates[0]['answer']

# Evaluate combined pipeline
r1_combined, rl_combined = evaluate_on_val(retrieve_and_rerank, "E5-ft + CE rerank")

make_submission(retrieve_and_rerank, "exp_overnight_reranked.csv",
    f"Phase 3: E5-ft (iterative HN) + cross-encoder reranker. Val ROUGE-1={r1_combined:.4f}")


# ============================================================
# PHASE 3b: PER-LANGUAGE RETRIEVAL + RERANK
# ============================================================
log("\n--- Phase 3b: Per-language retrieval ---")

# Build per-language FAISS indexes
lang_indexes = {}
for subset in combined['subset'].unique():
    mask = combined['subset'] == subset
    indices = np.where(mask.values)[0]
    sub_embeddings = corpus_r2[indices]
    sub_index = faiss.IndexFlatIP(sub_embeddings.shape[1])
    sub_index.add(sub_embeddings)
    lang_indexes[subset] = {'index': sub_index, 'indices': indices}
log(f"Built per-language indexes for {len(lang_indexes)} languages")

def retrieve_perlang_rerank(q_text, subset, top_k=20):
    """Per-language retrieval + cross-encoder rerank."""
    q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True).astype(np.float32)

    candidates = []

    # Same-language retrieval
    if subset in lang_indexes:
        li = lang_indexes[subset]
        D, I = li['index'].search(q_emb, min(top_k, li['index'].ntotal))
        for j in range(len(I[0])):
            real_idx = int(li['indices'][I[0][j]])
            cq = str(combined.iloc[real_idx]['input']).strip()
            if cq == q_text.strip():
                continue
            candidates.append({
                'answer': str(combined.iloc[real_idx]['output']),
                'e5_score': float(D[0][j]),
            })

    # Fall back to global if too few candidates
    if len(candidates) < 5:
        D, I = index_r2.search(q_emb, top_k)
        for j in range(top_k):
            idx = int(I[0][j])
            cq = str(combined.iloc[idx]['input']).strip()
            if cq == q_text.strip():
                continue
            ans = str(combined.iloc[idx]['output'])
            if not any(c['answer'] == ans for c in candidates):
                candidates.append({
                    'answer': ans,
                    'e5_score': float(D[0][j]),
                })

    if not candidates:
        return str(combined.iloc[0]['output'])

    # Rerank with cross-encoder
    ce_pairs = [[q_text, c['answer']] for c in candidates[:20]]
    ce_scores = ce_model.predict(ce_pairs, show_progress_bar=False)
    for i in range(len(ce_pairs)):
        candidates[i]['ce_score'] = float(ce_scores[i])
        candidates[i]['combined'] = 0.4 * candidates[i]['e5_score'] + 0.6 * candidates[i]['ce_score']

    candidates.sort(key=lambda x: -x['combined'])
    return candidates[0]['answer']

r1_perlang, rl_perlang = evaluate_on_val(retrieve_perlang_rerank, "Per-lang + CE rerank")

make_submission(retrieve_perlang_rerank, "exp_overnight_perlang_rerank.csv",
    f"Per-language E5-ft retrieval + CE reranker. Val ROUGE-1={r1_perlang:.4f}")


# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n" + "=" * 70)
log("🏆 OVERNIGHT TRAINING COMPLETE — FINAL SUMMARY")
log("=" * 70)
log(f"")
log(f"Baseline E5 (no fine-tuning):     Val ROUGE-1 = 0.5219")
log(f"E5 fine-tuned (exp13, no HN):     Val ROUGE-1 = 0.5580")
log(f"Round 1 (hard negatives):         Val ROUGE-1 = {r1_score_round1:.4f}")
log(f"Round 2 (iterative HN):           Val ROUGE-1 = {r1_score_round2:.4f}")
log(f"E5-ft + Cross-encoder rerank:     Val ROUGE-1 = {r1_combined:.4f}")
log(f"Per-lang + CE rerank:             Val ROUGE-1 = {r1_perlang:.4f}")
log(f"")
log(f"📥 DOWNLOAD ALL CSV FILES FROM /kaggle/working/ AND SUBMIT THE BEST ONE!")
log(f"")
log(f"Submissions saved:")
log(f"  1. exp_overnight_round1.csv")
log(f"  2. exp_overnight_round2.csv")
log(f"  3. exp_overnight_reranked.csv")
log(f"  4. exp_overnight_perlang_rerank.csv")
