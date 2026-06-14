"""
=============================================================================
OVERNIGHT v2: E5-LARGE + 3 Rounds Iterative Hard Negative Mining
=============================================================================
Run on Kaggle with GPU T4 x2. Designed to run UNATTENDED ~8 hours.

Cell 1: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm
Cell 2: paste this entire script
=============================================================================
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import torch
import faiss
import gc
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
# Try multiple possible paths (new account first)
POSSIBLE_PATHS = [
    Path('/kaggle/input/datasets/samuelkmwania/multilingual-health-qa-data/'),
    Path('/kaggle/input/multilingual-health-qa-data/'),
    Path('/kaggle/input/datasets/samuelmwania1/multilingual-health-qa-data/'),
    Path('data/raw/'),
]
DATA_DIR = None
for p in POSSIBLE_PATHS:
    if p.exists() and (p / 'Train.csv').exists():
        DATA_DIR = p
        break
if DATA_DIR is None:
    # Auto-detect: search /kaggle/input for Train.csv
    import glob
    found = glob.glob('/kaggle/input/**/Train.csv', recursive=True)
    if found:
        DATA_DIR = Path(found[0]).parent
    else:
        DATA_DIR = Path('data/raw/')
log(f"Using data from: {DATA_DIR}")
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

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)


def evaluate_on_val(model, index_obj, label=""):
    """Evaluate retrieval on val set."""
    val_qs = [f"query: {q}" for q in val_df['input'].fillna('').tolist()]
    val_emb = model.encode(val_qs, batch_size=32, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)
    r1_scores, rl_scores = [], []
    for idx in tqdm(range(len(val_df)), desc=f"Val [{label}]"):
        q = str(val_df.iloc[idx]['input']).strip()
        ref = str(val_df.iloc[idx]['output']).strip()
        q_emb = val_emb[idx:idx+1]
        D, I = index_obj.search(q_emb, 10)
        pred = ''
        for j in range(10):
            if str(combined.iloc[I[0][j]]['input']).strip() != q:
                pred = str(combined.iloc[I[0][j]]['output'])
                break
        if not pred:
            pred = str(combined.iloc[I[0][0]]['output'])
        r = scorer.score(ref, pred)
        r1_scores.append(r['rouge1'].fmeasure)
        rl_scores.append(r['rougeL'].fmeasure)
    r1, rl = np.mean(r1_scores), np.mean(rl_scores)
    log(f"[{label}] Val ROUGE-1: {r1:.4f} | ROUGE-L: {rl:.4f}")
    return r1, rl


def make_submission(model, index_obj, filename, comment):
    """Generate submission CSV."""
    test_qs = [f"query: {q}" for q in test_df['input'].fillna('').tolist()]
    test_emb = model.encode(test_qs, batch_size=32, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)
    rows = []
    for idx in tqdm(range(len(test_df)), desc=f"Sub [{filename}]"):
        q_emb = test_emb[idx:idx+1]
        D, I = index_obj.search(q_emb, 3)
        answer = str(combined.iloc[I[0][0]]['output'])
        rows.append({
            'ID': test_df.iloc[idx]['ID'],
            'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
        })
    sub = pd.DataFrame(rows)
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    path = OUTPUT_DIR / filename
    sub.to_csv(path, index=False)
    log(f"✅ Saved: {path} | {comment}")
    return path


def encode_corpus(model):
    """Encode corpus and build FAISS index."""
    corpus_qs = [f"query: {q}" for q in questions_raw]
    emb = model.encode(corpus_qs, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return emb, idx


def mine_hard_negatives(corpus_emb, faiss_index, max_neg=2):
    """Mine hard negatives: similar questions with different answers."""
    from sentence_transformers import InputExample
    examples = []
    for i in tqdm(range(len(combined)), desc="Mining HN"):
        q = questions_raw[i]
        a = answers_raw[i]
        if not q.strip() or not a.strip():
            continue
        q_emb = corpus_emb[i:i+1]
        D, I = faiss_index.search(q_emb, 30)
        hard_negs = []
        for j in range(30):
            idx = int(I[0][j])
            if idx == i:
                continue
            cand_a = answers_raw[idx]
            if cand_a.strip() != a.strip() and len(hard_negs) < max_neg:
                hard_negs.append(f"passage: {cand_a}")
        if hard_negs:
            texts = [f"query: {q}", f"passage: {a}"] + hard_negs
        else:
            texts = [f"query: {q}", f"passage: {a}"]
        examples.append(InputExample(texts=texts))
    return examples


# ============================================================
# PHASE 1: E5-LARGE + ITERATIVE HARD NEGATIVE MINING
# ============================================================
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

log("=" * 70)
log("E5-LARGE + ITERATIVE HARD NEGATIVE MINING")
log("=" * 70)

log("\nLoading E5-LARGE model...")
model = SentenceTransformer('intfloat/multilingual-e5-large', device='cuda:0')
log(f"Model loaded! Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

best_r1 = 0
best_filename = ""

# ============================================================
# ROUND 1: 3 epochs
# ============================================================
log("\n--- ROUND 1 ---")
corpus_emb, faiss_idx = encode_corpus(model)
examples = mine_hard_negatives(corpus_emb, faiss_idx, max_neg=2)
log(f"Training examples: {len(examples)}")

del corpus_emb
gc.collect()
torch.cuda.empty_cache()

train_loss = losses.MultipleNegativesRankingLoss(model)
loader = DataLoader(examples, shuffle=True, batch_size=4)
log(f"Round 1: 3 epochs, batch_size=4, steps={len(loader)*3}")

model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=3,
    warmup_steps=200,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5large-r1'),
    use_amp=True,
)
log("✅ Round 1 complete!")

del examples, loader, train_loss
gc.collect()
torch.cuda.empty_cache()

corpus_emb, faiss_idx = encode_corpus(model)
r1, _ = evaluate_on_val(model, faiss_idx, "E5-large R1")
make_submission(model, faiss_idx, "exp_e5large_round1.csv",
    f"E5-large + HN mining R1. Val={r1:.4f}")
if r1 > best_r1:
    best_r1 = r1
    best_filename = "exp_e5large_round1.csv"
model.save(str(OUTPUT_DIR / 'e5large-r1-model'))

# ============================================================
# ROUND 2: 2 epochs
# ============================================================
log("\n--- ROUND 2 ---")
examples = mine_hard_negatives(corpus_emb, faiss_idx, max_neg=3)
log(f"Training examples: {len(examples)}")

del corpus_emb
gc.collect()
torch.cuda.empty_cache()

train_loss = losses.MultipleNegativesRankingLoss(model)
loader = DataLoader(examples, shuffle=True, batch_size=4)
log(f"Round 2: 2 epochs, batch_size=4, steps={len(loader)*2}")

model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=2,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5large-r2'),
    use_amp=True,
)
log("✅ Round 2 complete!")

del examples, loader, train_loss
gc.collect()
torch.cuda.empty_cache()

corpus_emb, faiss_idx = encode_corpus(model)
r1, _ = evaluate_on_val(model, faiss_idx, "E5-large R2")
make_submission(model, faiss_idx, "exp_e5large_round2.csv",
    f"E5-large + iterative HN R2. Val={r1:.4f}")
if r1 > best_r1:
    best_r1 = r1
    best_filename = "exp_e5large_round2.csv"
model.save(str(OUTPUT_DIR / 'e5large-r2-model'))

# ============================================================
# ROUND 3: 2 epochs with Q-to-Q pairs mixed in
# ============================================================
log("\n--- ROUND 3: Question-to-Question training ---")

# Mine hard negatives (question→answer)
qa_examples = mine_hard_negatives(corpus_emb, faiss_idx, max_neg=3)

# Also create question→question pairs
log("Creating question-to-question pairs...")
qq_examples = []
# Group questions by answer (same answer = positive pair)
from collections import defaultdict
answer_to_qidx = defaultdict(list)
for i, a in enumerate(answers_raw):
    a_key = a.strip()[:200]  # use first 200 chars as key
    if a_key:
        answer_to_qidx[a_key].append(i)

for a_key, indices in tqdm(answer_to_qidx.items(), desc="Q-Q pairs"):
    if len(indices) < 2:
        continue
    for i in range(min(3, len(indices))):  # limit pairs per answer
        anchor_idx = indices[i]
        pos_idx = indices[(i + 1) % len(indices)]
        q_anchor = questions_raw[anchor_idx]
        q_pos = questions_raw[pos_idx]
        if q_anchor.strip() == q_pos.strip():
            continue
        # Find a hard negative question (different answer)
        q_emb = corpus_emb[anchor_idx:anchor_idx+1]
        D, I = faiss_idx.search(q_emb, 15)
        neg_q = None
        for j in range(15):
            nidx = int(I[0][j])
            if nidx == anchor_idx or nidx == pos_idx:
                continue
            if answers_raw[nidx].strip()[:200] != a_key:
                neg_q = questions_raw[nidx]
                break
        if neg_q:
            qq_examples.append(InputExample(
                texts=[f"query: {q_anchor}", f"query: {q_pos}", f"query: {neg_q}"]
            ))

log(f"Q-A examples: {len(qa_examples)}, Q-Q examples: {len(qq_examples)}")
all_examples = qa_examples + qq_examples[:len(qa_examples)]  # balance
np.random.shuffle(all_examples)

del corpus_emb, qa_examples, qq_examples
gc.collect()
torch.cuda.empty_cache()

train_loss = losses.MultipleNegativesRankingLoss(model)
loader = DataLoader(all_examples, shuffle=True, batch_size=4)
log(f"Round 3: 2 epochs, batch_size=4, steps={len(loader)*2}")

model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=2,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5large-r3'),
    use_amp=True,
)
log("✅ Round 3 complete!")

del all_examples, loader, train_loss
gc.collect()
torch.cuda.empty_cache()

corpus_emb, faiss_idx = encode_corpus(model)
r1, _ = evaluate_on_val(model, faiss_idx, "E5-large R3 (QA+QQ)")
make_submission(model, faiss_idx, "exp_e5large_round3.csv",
    f"E5-large + 3-round HN + Q-Q training. Val={r1:.4f}")
if r1 > best_r1:
    best_r1 = r1
    best_filename = "exp_e5large_round3.csv"
model.save(str(OUTPUT_DIR / 'e5large-final'))

# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n" + "=" * 70)
log("🏆 OVERNIGHT v2 COMPLETE — FINAL SUMMARY")
log("=" * 70)
log(f"")
log(f"Previous best (E5-base iterHN):   Val ROUGE-1 = 0.6045 → LB 0.6410")
log(f"E5-LARGE Round 1:                 Val ROUGE-1 = see above")
log(f"E5-LARGE Round 2:                 Val ROUGE-1 = see above")
log(f"E5-LARGE Round 3 (QA+QQ):         Val ROUGE-1 = see above")
log(f"")
log(f"🏆 BEST: {best_filename} (Val ROUGE-1 = {best_r1:.4f})")
log(f"")
log(f"📥 DOWNLOAD {best_filename} AND SUBMIT!")
