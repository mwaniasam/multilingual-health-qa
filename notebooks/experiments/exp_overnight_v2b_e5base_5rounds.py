"""
=============================================================================
OVERNIGHT v2b: E5-BASE + 5 Rounds Iterative Hard Negative Mining
=============================================================================
The SAFE approach: same proven E5-base model, but 5 rounds of iterative
hard negative mining instead of 2. Each round sharpens the model further.

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

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def encode_corpus(model):
    corpus_qs = [f"query: {q}" for q in questions_raw]
    emb = model.encode(corpus_qs, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return emb, idx


def evaluate_on_val(model, index_obj, label=""):
    val_qs = [f"query: {q}" for q in val_df['input'].fillna('').tolist()]
    val_emb = model.encode(val_qs, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)
    r1_scores = []
    for idx in range(len(val_df)):
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
    r1 = np.mean(r1_scores)
    log(f"[{label}] Val ROUGE-1: {r1:.4f}")
    return r1


def make_submission(model, index_obj, filename, comment):
    test_qs = [f"query: {q}" for q in test_df['input'].fillna('').tolist()]
    test_emb = model.encode(test_qs, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)
    rows = []
    for idx in range(len(test_df)):
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
    sub.to_csv(OUTPUT_DIR / filename, index=False)
    log(f"✅ Saved: {filename} | {comment}")


def mine_hn(corpus_emb, faiss_idx, max_neg=3):
    examples = []
    for i in tqdm(range(len(combined)), desc="Mining HN"):
        q, a = questions_raw[i], answers_raw[i]
        if not q.strip() or not a.strip():
            continue
        D, I = faiss_idx.search(corpus_emb[i:i+1], 30)
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


# ============================================================
# 5 ROUNDS OF ITERATIVE HARD NEGATIVE MINING
# ============================================================
log("=" * 70)
log("E5-BASE + 5 ROUNDS ITERATIVE HARD NEGATIVE MINING")
log("=" * 70)

model = SentenceTransformer('intfloat/multilingual-e5-base', device='cuda:0')
best_r1 = 0
best_file = ""

configs = [
    # (round, epochs, max_neg, batch_size)
    (1, 5, 2, 8),
    (2, 3, 3, 8),
    (3, 2, 4, 8),
    (4, 2, 5, 8),
    (5, 1, 5, 8),
]

for rnd, epochs, max_neg, bs in configs:
    log(f"\n--- ROUND {rnd}: {epochs} epochs, max_neg={max_neg}, bs={bs} ---")

    corpus_emb, faiss_idx = encode_corpus(model)
    examples = mine_hn(corpus_emb, faiss_idx, max_neg=max_neg)
    log(f"Examples: {len(examples)}")

    del corpus_emb
    gc.collect()
    torch.cuda.empty_cache()

    train_loss = losses.MultipleNegativesRankingLoss(model)
    loader = DataLoader(examples, shuffle=True, batch_size=bs)
    log(f"Steps: {len(loader) * epochs}")

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=min(200, len(loader)),
        show_progress_bar=True,
        output_path=str(OUTPUT_DIR / f'e5base-r{rnd}'),
        use_amp=True,
    )
    log(f"✅ Round {rnd} training complete!")

    del examples, loader, train_loss
    gc.collect()
    torch.cuda.empty_cache()

    corpus_emb, faiss_idx = encode_corpus(model)
    r1 = evaluate_on_val(model, faiss_idx, f"R{rnd}")
    fname = f"exp_e5base_5round_r{rnd}.csv"
    make_submission(model, faiss_idx, fname, f"E5-base R{rnd}/{5}. Val={r1:.4f}")

    if r1 > best_r1:
        best_r1 = r1
        best_file = fname
    model.save(str(OUTPUT_DIR / f'e5base-r{rnd}-model'))

# ============================================================
# BONUS: Add question-to-question pairs for final round
# ============================================================
log("\n--- BONUS ROUND: Q-to-Q training ---")
corpus_emb, faiss_idx = encode_corpus(model)

# Create Q-Q pairs from questions sharing same answer
answer_to_qidx = defaultdict(list)
for i, a in enumerate(answers_raw):
    key = a.strip()[:200]
    if key:
        answer_to_qidx[key].append(i)

qq_examples = []
for key, indices in tqdm(answer_to_qidx.items(), desc="Q-Q pairs"):
    if len(indices) < 2:
        continue
    for i in range(min(3, len(indices))):
        ai = indices[i]
        pi = indices[(i+1) % len(indices)]
        if questions_raw[ai].strip() == questions_raw[pi].strip():
            continue
        D, I = faiss_idx.search(corpus_emb[ai:ai+1], 15)
        neg_q = None
        for j in range(15):
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

log(f"Q-Q examples: {len(qq_examples)}")

del corpus_emb
gc.collect()
torch.cuda.empty_cache()

if qq_examples:
    train_loss = losses.MultipleNegativesRankingLoss(model)
    loader = DataLoader(qq_examples, shuffle=True, batch_size=8)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True,
        output_path=str(OUTPUT_DIR / 'e5base-qq'),
        use_amp=True,
    )
    log("✅ Q-Q round complete!")
    del qq_examples, loader, train_loss
    gc.collect()
    torch.cuda.empty_cache()

    corpus_emb, faiss_idx = encode_corpus(model)
    r1 = evaluate_on_val(model, faiss_idx, "R5+QQ")
    fname = "exp_e5base_5round_qq.csv"
    make_submission(model, faiss_idx, fname, f"E5-base 5-round HN + Q-Q. Val={r1:.4f}")
    if r1 > best_r1:
        best_r1 = r1
        best_file = fname
    model.save(str(OUTPUT_DIR / 'e5base-final'))

# ============================================================
log("\n" + "=" * 70)
log("🏆 OVERNIGHT v2b COMPLETE")
log("=" * 70)
log(f"Previous best: Val=0.6045 → LB=0.6410")
log(f"🏆 BEST THIS RUN: {best_file} (Val ROUGE-1 = {best_r1:.4f})")
log(f"📥 DOWNLOAD {best_file} AND SUBMIT!")
