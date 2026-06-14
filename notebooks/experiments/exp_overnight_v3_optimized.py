"""
=============================================================================
OVERNIGHT v3: E5-LARGE — OPTIMIZED (2 HN rounds + HEAVY Q-Q training)
=============================================================================
Based on findings: HN plateaus after 2 rounds, Q-Q training is the real gain.

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
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================================
# CONFIG — auto-detect data path
# ============================================================
POSSIBLE_PATHS = [
    Path('/kaggle/input/datasets/samuelmwania1/multilingual-health-qa-data/'),
    Path('/kaggle/input/datasets/samuelkmwania/multilingual-health-qa-data/'),
    Path('/kaggle/input/multilingual-health-qa-data/'),
]
DATA_DIR = None
for p in POSSIBLE_PATHS:
    if p.exists() and (p / 'Train.csv').exists():
        DATA_DIR = p
        break
if DATA_DIR is None:
    import glob
    found = glob.glob('/kaggle/input/**/Train.csv', recursive=True)
    DATA_DIR = Path(found[0]).parent if found else Path('data/raw/')
log(f"Data: {DATA_DIR}")

OUTPUT_DIR = Path('/kaggle/working/')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path('submissions/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def encode_corpus(mdl):
    emb = mdl.encode([f"query: {q}" for q in questions_raw],
                     batch_size=32, show_progress_bar=True,
                     normalize_embeddings=True).astype(np.float32)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return emb, idx


def evaluate_val(mdl, fidx, label=""):
    val_qs = [f"query: {q}" for q in val_df['input'].fillna('').tolist()]
    val_emb = mdl.encode(val_qs, batch_size=32, show_progress_bar=True,
                         normalize_embeddings=True).astype(np.float32)
    r1s = []
    for i in range(len(val_df)):
        q = str(val_df.iloc[i]['input']).strip()
        ref = str(val_df.iloc[i]['output']).strip()
        D, I = fidx.search(val_emb[i:i+1], 10)
        pred = ''
        for j in range(10):
            if str(combined.iloc[I[0][j]]['input']).strip() != q:
                pred = str(combined.iloc[I[0][j]]['output'])
                break
        if not pred:
            pred = str(combined.iloc[I[0][0]]['output'])
        r1s.append(scorer.score(ref, pred)['rouge1'].fmeasure)
    r1 = np.mean(r1s)
    log(f"[{label}] Val ROUGE-1: {r1:.4f}")
    return r1


def save_submission(mdl, fidx, fname, comment):
    test_qs = [f"query: {q}" for q in test_df['input'].fillna('').tolist()]
    test_emb = mdl.encode(test_qs, batch_size=32, show_progress_bar=True,
                          normalize_embeddings=True).astype(np.float32)
    rows = []
    for i in range(len(test_df)):
        D, I = fidx.search(test_emb[i:i+1], 3)
        rows.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': str(combined.iloc[I[0][0]]['output']),
            'TargetR1F1': str(combined.iloc[I[0][0]]['output']),
            'TargetLLM': str(combined.iloc[I[0][0]]['output']),
        })
    sub = pd.DataFrame(rows)
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    sub.to_csv(OUTPUT_DIR / fname, index=False)
    log(f"✅ Saved: {fname} | {comment}")


def mine_hn(corpus_emb, fidx, max_neg=2):
    examples = []
    for i in tqdm(range(len(combined)), desc="Mining HN"):
        q, a = questions_raw[i], answers_raw[i]
        if not q.strip() or not a.strip():
            continue
        D, I = fidx.search(corpus_emb[i:i+1], 30)
        negs = []
        for j in range(30):
            idx = int(I[0][j])
            if idx == i:
                continue
            if answers_raw[idx].strip() != a.strip() and len(negs) < max_neg:
                negs.append(f"passage: {answers_raw[idx]}")
        texts = [f"query: {q}", f"passage: {a}"] + negs if negs else [f"query: {q}", f"passage: {a}"]
        examples.append(InputExample(texts=texts))
    return examples


def build_qq_pairs(corpus_emb, fidx, max_pairs_per_answer=5):
    """Build question-to-question pairs from questions sharing similar answers."""
    log("Building Q-Q pairs...")

    # Group by answer (exact match)
    answer_groups = defaultdict(list)
    for i, a in enumerate(answers_raw):
        key = a.strip()[:200]
        if key:
            answer_groups[key].append(i)

    qq_examples = []

    # Type 1: Same-answer Q-Q pairs (strongest signal)
    for key, indices in tqdm(answer_groups.items(), desc="Same-answer Q-Q"):
        if len(indices) < 2:
            continue
        pairs_made = 0
        for i in range(len(indices)):
            if pairs_made >= max_pairs_per_answer:
                break
            ai = indices[i]
            pi = indices[(i + 1) % len(indices)]
            if questions_raw[ai].strip() == questions_raw[pi].strip():
                continue
            # Find hard negative question
            D, I = fidx.search(corpus_emb[ai:ai+1], 20)
            neg_q = None
            for j in range(20):
                ni = int(I[0][j])
                if ni == ai or ni == pi:
                    continue
                if answers_raw[ni].strip()[:200] != key:
                    neg_q = questions_raw[ni]
                    break
            if neg_q:
                qq_examples.append(InputExample(
                    texts=[f"query: {questions_raw[ai]}", f"query: {questions_raw[pi]}", f"query: {neg_q}"]
                ))
                pairs_made += 1

    # Type 2: Cross-language Q-Q pairs (questions about same topic in different languages)
    log("Building cross-language Q-Q pairs...")
    for key, indices in tqdm(answer_groups.items(), desc="Cross-lang Q-Q"):
        if len(indices) < 2:
            continue
        # Group by language within this answer group
        lang_groups = defaultdict(list)
        for idx in indices:
            lang_groups[subsets_raw[idx]].append(idx)
        # Pair questions across languages
        langs = list(lang_groups.keys())
        pairs_made = 0
        for li in range(len(langs)):
            for lj in range(li + 1, len(langs)):
                if pairs_made >= 3:
                    break
                idx_a = lang_groups[langs[li]][0]
                idx_b = lang_groups[langs[lj]][0]
                if questions_raw[idx_a].strip() == questions_raw[idx_b].strip():
                    continue
                # Negative from different answer
                D, I = fidx.search(corpus_emb[idx_a:idx_a+1], 20)
                neg_q = None
                for j in range(20):
                    ni = int(I[0][j])
                    if ni == idx_a or ni == idx_b:
                        continue
                    if answers_raw[ni].strip()[:200] != key:
                        neg_q = questions_raw[ni]
                        break
                if neg_q:
                    qq_examples.append(InputExample(
                        texts=[f"query: {questions_raw[idx_a]}", f"query: {questions_raw[idx_b]}", f"query: {neg_q}"]
                    ))
                    pairs_made += 1

    log(f"Total Q-Q pairs: {len(qq_examples)}")
    return qq_examples


def do_training(mdl, examples, epochs, bs, label, warmup=100):
    """Train and clean up."""
    train_loss = losses.MultipleNegativesRankingLoss(mdl)
    loader = DataLoader(examples, shuffle=True, batch_size=bs)
    log(f"[{label}] {len(examples)} examples, {epochs} epochs, bs={bs}, steps={len(loader)*epochs}")
    mdl.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
        output_path=str(OUTPUT_DIR / label),
        use_amp=True,
    )
    log(f"✅ [{label}] Training complete!")
    del loader, train_loss
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# LOAD E5-LARGE
# ============================================================
log("=" * 70)
log("E5-LARGE: OPTIMIZED PIPELINE")
log("=" * 70)
model = SentenceTransformer('intfloat/multilingual-e5-large', device='cuda:0')
log(f"Loaded! {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

best_r1 = 0
best_file = ""

# ============================================================
# PHASE 1: 2 ROUNDS HN MINING (proven, don't overdo)
# ============================================================
log("\n" + "=" * 50)
log("PHASE 1: Hard Negative Mining (2 rounds only)")
log("=" * 50)

# Round 1
corpus_emb, fidx = encode_corpus(model)
examples = mine_hn(corpus_emb, fidx, max_neg=2)
del corpus_emb; gc.collect(); torch.cuda.empty_cache()
do_training(model, examples, epochs=3, bs=4, label="e5l-hn-r1", warmup=200)
del examples; gc.collect(); torch.cuda.empty_cache()

corpus_emb, fidx = encode_corpus(model)
r1 = evaluate_val(model, fidx, "E5L HN-R1")
save_submission(model, fidx, "exp_e5l_hn_r1.csv", f"E5-large HN R1. Val={r1:.4f}")
if r1 > best_r1: best_r1, best_file = r1, "exp_e5l_hn_r1.csv"

# Round 2
examples = mine_hn(corpus_emb, fidx, max_neg=3)
del corpus_emb; gc.collect(); torch.cuda.empty_cache()
do_training(model, examples, epochs=2, bs=4, label="e5l-hn-r2")
del examples; gc.collect(); torch.cuda.empty_cache()

corpus_emb, fidx = encode_corpus(model)
r1 = evaluate_val(model, fidx, "E5L HN-R2")
save_submission(model, fidx, "exp_e5l_hn_r2.csv", f"E5-large HN R2. Val={r1:.4f}")
if r1 > best_r1: best_r1, best_file = r1, "exp_e5l_hn_r2.csv"

# ============================================================
# PHASE 2: HEAVY Q-Q TRAINING (the breakthrough technique)
# ============================================================
log("\n" + "=" * 50)
log("PHASE 2: Q-to-Q Training (THE breakthrough)")
log("=" * 50)

# Q-Q Round 1: same-answer + cross-language pairs
qq_examples = build_qq_pairs(corpus_emb, fidx, max_pairs_per_answer=5)
del corpus_emb; gc.collect(); torch.cuda.empty_cache()
do_training(model, qq_examples, epochs=3, bs=4, label="e5l-qq-r1")
del qq_examples; gc.collect(); torch.cuda.empty_cache()

corpus_emb, fidx = encode_corpus(model)
r1 = evaluate_val(model, fidx, "E5L QQ-R1")
save_submission(model, fidx, "exp_e5l_qq_r1.csv", f"E5-large HN+QQ R1. Val={r1:.4f}")
if r1 > best_r1: best_r1, best_file = r1, "exp_e5l_qq_r1.csv"

# Q-Q Round 2: re-mine Q-Q pairs with improved model
qq_examples2 = build_qq_pairs(corpus_emb, fidx, max_pairs_per_answer=8)
del corpus_emb; gc.collect(); torch.cuda.empty_cache()
do_training(model, qq_examples2, epochs=2, bs=4, label="e5l-qq-r2")
del qq_examples2; gc.collect(); torch.cuda.empty_cache()

corpus_emb, fidx = encode_corpus(model)
r1 = evaluate_val(model, fidx, "E5L QQ-R2")
save_submission(model, fidx, "exp_e5l_qq_r2.csv", f"E5-large HN+QQ R2. Val={r1:.4f}")
if r1 > best_r1: best_r1, best_file = r1, "exp_e5l_qq_r2.csv"

# ============================================================
# PHASE 3: FINAL COMBINED ROUND (mix Q-A + Q-Q)
# ============================================================
log("\n" + "=" * 50)
log("PHASE 3: Final combined Q-A + Q-Q training")
log("=" * 50)

# Mine fresh HN for Q-A pairs
qa_examples = mine_hn(corpus_emb, fidx, max_neg=2)
qq_examples3 = build_qq_pairs(corpus_emb, fidx, max_pairs_per_answer=5)
del corpus_emb; gc.collect(); torch.cuda.empty_cache()

# Combine and shuffle
all_examples = qa_examples + qq_examples3
np.random.seed(42)
np.random.shuffle(all_examples)
log(f"Combined: {len(qa_examples)} Q-A + {len(qq_examples3)} Q-Q = {len(all_examples)} total")
del qa_examples, qq_examples3

do_training(model, all_examples, epochs=1, bs=4, label="e5l-final")
del all_examples; gc.collect(); torch.cuda.empty_cache()

corpus_emb, fidx = encode_corpus(model)
r1 = evaluate_val(model, fidx, "E5L FINAL")
save_submission(model, fidx, "exp_e5l_final.csv", f"E5-large full pipeline. Val={r1:.4f}")
if r1 > best_r1: best_r1, best_file = r1, "exp_e5l_final.csv"

model.save(str(OUTPUT_DIR / 'e5large-final-model'))

# ============================================================
log("\n" + "=" * 70)
log("🏆 E5-LARGE PIPELINE COMPLETE")
log("=" * 70)
log(f"")
log(f"Previous best (E5-base + QQ):  Val = 0.6114 → LB = ???")
log(f"Previous best (E5-base HN):    Val = 0.6045 → LB = 0.6410")
log(f"")
log(f"🏆 BEST THIS RUN: {best_file} (Val ROUGE-1 = {best_r1:.4f})")
log(f"📥 DOWNLOAD {best_file} AND SUBMIT!")
