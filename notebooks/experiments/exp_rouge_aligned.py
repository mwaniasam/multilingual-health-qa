"""
=============================================================================
ROUGE-ALIGNED FINE-TUNING of AfriE5
=============================================================================
WHY this works:
- Current AfriE5: ranks by question similarity
- Problem: similar question ≠ similar answer wording
- Fix: teach AfriE5 "this question is the right match because its ANSWER
  has the highest ROUGE overlap with the reference"
- This directly aligns retrieval with the evaluation metric

Runs on T4 (16GB). ~2 hours total.

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm
Cell 3: Paste this
=============================================================================
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import faiss
import gc
import re
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# PATHS
# ============================================================
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
    else:
        log("Google Drive already mounted.")
    DATA_DIR = Path('/content/drive/MyDrive/multilingual-health-qa/data')
    OUTPUT_DIR = Path('/content/drive/MyDrive/multilingual-health-qa/outputs')
    AFRIE5_DIR = Path('/content/drive/MyDrive/multilingual-health-qa/outputs/afrie5-final-model')
except ImportError:
    DATA_DIR = Path('data/raw/')
    OUTPUT_DIR = Path('outputs/')
    AFRIE5_DIR = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
log("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df   = pd.read_csv(DATA_DIR / 'Val.csv')
test_df  = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    log(f"  {name}: {len(df)} rows")

combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])
combined = combined.reset_index(drop=True)
questions_raw = combined['input'].fillna('').astype(str).tolist()
answers_raw   = combined['output'].fillna('').astype(str).tolist()
log(f"Combined corpus: {len(combined)} samples")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# FAST ROUGE (100x faster than rouge_score for data building)
# ============================================================
def fast_rouge1_f1(ref_str, hyp_str):
    """Fast ROUGE-1 F1 using set overlap. ~100x faster than rouge_score."""
    ref_tokens = set(ref_str.lower().split())
    hyp_tokens = set(hyp_str.lower().split())
    if not ref_tokens or not hyp_tokens:
        return 0.0
    overlap = len(ref_tokens & hyp_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

# ============================================================
# STEP 1: LOAD MODEL + BUILD INDEX
# ============================================================
log(f"\n{'='*60}")
log("STEP 1: Load AfriE5 + Build index")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

PREFIX = "query: "

if AFRIE5_DIR and AFRIE5_DIR.exists():
    bienc = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0')
    log(f"AfriE5 loaded from Drive: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")
else:
    bienc = SentenceTransformer('McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
    log(f"AfriE5 from HuggingFace: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

log("Encoding corpus questions...")
corpus_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
fidx = faiss.IndexFlatIP(corpus_emb.shape[1])
fidx.add(corpus_emb)
log(f"Index: {corpus_emb.shape}")

# ============================================================
# STEP 2: MEASURE CURRENT BASELINE ON VAL
# ============================================================
log(f"\n{'='*60}")
log("STEP 2: Current baseline on val")
log(f"{'='*60}")

val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

baseline_r1s, baseline_rls = [], []
per_lang_base = defaultdict(lambda: {'r1': [], 'rl': []})

for i in tqdm(range(len(val_df)), desc="Baseline eval"):
    q = str(val_df.iloc[i]['input']).strip()
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref: continue

    D, I = fidx.search(val_emb[i:i+1], 5)
    answer = ''
    for j in range(5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        answer = str(combined.iloc[ci]['output'])
        break
    if not answer:
        answer = str(combined.iloc[int(I[0][0])]['output'])

    r = scorer.score(ref, answer)
    baseline_r1s.append(r['rouge1'].fmeasure)
    baseline_rls.append(r['rougeL'].fmeasure)
    per_lang_base[sub]['r1'].append(r['rouge1'].fmeasure)
    per_lang_base[sub]['rl'].append(r['rougeL'].fmeasure)

b_r1, b_rl = np.mean(baseline_r1s), np.mean(baseline_rls)
log(f"\nBaseline: R1={b_r1:.4f} RL={b_rl:.4f}")
for sub in sorted(per_lang_base.keys()):
    d = per_lang_base[sub]
    log(f"  {sub:<12} R1={np.mean(d['r1']):.4f} RL={np.mean(d['rl']):.4f}")

# ============================================================
# STEP 3: BUILD ROUGE-ALIGNED TRAINING DATA
# ============================================================
log(f"\n{'='*60}")
log("STEP 3: Build ROUGE-aligned training data")
log(f"{'='*60}")
log("For each train Q, finding which top-30 candidate has the")
log("BEST ROUGE overlap → that becomes the positive pair.")

train_qs_text = train_df['input'].fillna('').astype(str).tolist()
train_as_text = train_df['output'].fillna('').astype(str).tolist()

log("Encoding train questions...")
train_q_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in train_qs_text],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

CAND_K = 30  # candidates per query
rouge_aligned_pairs = []
improvements = 0
total_checked = 0

for i in tqdm(range(len(train_df)), desc="ROUGE-aligned pairs"):
    q = train_qs_text[i].strip()
    ref = train_as_text[i].strip()
    if not q or not ref:
        continue

    D, I = fidx.search(train_q_emb[i:i+1], CAND_K + 5)

    # Get top-1 (current best) and all candidates
    candidates = []
    for j in range(CAND_K + 5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        cq = str(combined.iloc[ci]['input']).strip()
        if cq == q: continue  # skip self
        ca = str(combined.iloc[ci]['output']).strip()
        candidates.append((ci, cq, ca))
        if len(candidates) >= CAND_K: break

    if len(candidates) < 3:
        continue

    # Compute fast ROUGE for each candidate's answer vs reference
    rouge_scores = []
    for ci, cq, ca in candidates:
        r1 = fast_rouge1_f1(ref, ca)
        rouge_scores.append(r1)

    # Current top-1 ROUGE
    top1_rouge = rouge_scores[0]

    # Best ROUGE candidate
    best_idx = int(np.argmax(rouge_scores))
    best_rouge = rouge_scores[best_idx]
    best_ci, best_cq, best_ca = candidates[best_idx]

    total_checked += 1

    # Only create training pair if the best candidate is NOT already top-1
    # AND the improvement is meaningful (> 0.05 ROUGE)
    if best_idx > 0 and best_rouge - top1_rouge > 0.05:
        # Positive pair: query → best ROUGE question
        rouge_aligned_pairs.append(InputExample(
            texts=[f"{PREFIX}{q}", f"{PREFIX}{best_cq}"]
        ))
        improvements += 1

    # Also add the top-1 as positive (to maintain what's already correct)
    top1_ci, top1_cq, top1_ca = candidates[0]
    if top1_rouge > 0.3:  # only if top-1 is decent
        rouge_aligned_pairs.append(InputExample(
            texts=[f"{PREFIX}{q}", f"{PREFIX}{top1_cq}"]
        ))

log(f"\nROUGE-aligned training examples: {len(rouge_aligned_pairs)}")
log(f"Cases where current top-1 is NOT the best: {improvements}/{total_checked} "
    f"({100*improvements/max(total_checked,1):.1f}%)")

# Free embeddings
del train_q_emb
gc.collect()

# ============================================================
# STEP 4: FINE-TUNE AfriE5 WITH ROUGE-ALIGNED PAIRS
# ============================================================
log(f"\n{'='*60}")
log("STEP 4: Fine-tune AfriE5 with ROUGE-aligned objective")
log(f"{'='*60}")

train_loader = DataLoader(rouge_aligned_pairs, batch_size=16, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(bienc)

# Warmup steps
total_steps = len(train_loader) * 1  # 1 epoch
warmup_steps = int(total_steps * 0.1)

log(f"Training examples: {len(rouge_aligned_pairs)}")
log(f"Batch size: 16")
log(f"Total steps: {total_steps}")
log(f"Warmup: {warmup_steps}")
log(f"Learning rate: 1e-5 (low — preserving existing knowledge)")

bienc.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=1,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': 1e-5},
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'afrie5-rouge-aligned'),
)

log("Training complete!")
bienc.save(str(OUTPUT_DIR / 'afrie5-rouge-aligned'))
log("Model saved to Drive!")

# ============================================================
# STEP 5: EVALUATE IMPROVED MODEL ON VAL
# ============================================================
log(f"\n{'='*60}")
log("STEP 5: Evaluate ROUGE-aligned model on val")
log(f"{'='*60}")

# Re-encode everything with the improved model
log("Re-encoding corpus...")
corpus_emb_new = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
fidx_new = faiss.IndexFlatIP(corpus_emb_new.shape[1])
fidx_new.add(corpus_emb_new)

log("Re-encoding val...")
val_emb_new = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

new_r1s, new_rls = [], []
per_lang_new = defaultdict(lambda: {'r1': [], 'rl': []})

for i in tqdm(range(len(val_df)), desc="New model eval"):
    q = str(val_df.iloc[i]['input']).strip()
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref: continue

    D, I = fidx_new.search(val_emb_new[i:i+1], 5)
    answer = ''
    for j in range(5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        answer = str(combined.iloc[ci]['output'])
        break
    if not answer:
        answer = str(combined.iloc[int(I[0][0])]['output'])

    r = scorer.score(ref, answer)
    new_r1s.append(r['rouge1'].fmeasure)
    new_rls.append(r['rougeL'].fmeasure)
    per_lang_new[sub]['r1'].append(r['rouge1'].fmeasure)
    per_lang_new[sub]['rl'].append(r['rougeL'].fmeasure)

n_r1, n_rl = np.mean(new_r1s), np.mean(new_rls)

log(f"\n{'='*60}")
log(f"{'Method':<30} {'ROUGE-1':>10} {'ROUGE-L':>10}")
log(f"{'-'*52}")
log(f"{'Before (baseline)':30} {b_r1:>10.4f} {b_rl:>10.4f}")
log(f"{'After (ROUGE-aligned)':30} {n_r1:>10.4f} {n_rl:>10.4f}")
log(f"{'IMPROVEMENT':30} {n_r1-b_r1:>+10.4f} {n_rl-b_rl:>+10.4f}")
log(f"{'='*60}")

log(f"\nPer-language comparison:")
log(f"{'Subset':<12} {'Base R1':>8} {'New R1':>8} {'Δ':>7} | {'Base RL':>8} {'New RL':>8} {'Δ':>7}")
for sub in sorted(set(list(per_lang_base.keys()) + list(per_lang_new.keys()))):
    br1 = np.mean(per_lang_base[sub]['r1']) if per_lang_base[sub]['r1'] else 0
    brl = np.mean(per_lang_base[sub]['rl']) if per_lang_base[sub]['rl'] else 0
    nr1 = np.mean(per_lang_new[sub]['r1']) if per_lang_new[sub]['r1'] else 0
    nrl = np.mean(per_lang_new[sub]['rl']) if per_lang_new[sub]['rl'] else 0
    log(f"  {sub:<12} {br1:>8.4f} {nr1:>8.4f} {nr1-br1:>+7.4f} | {brl:>8.4f} {nrl:>8.4f} {nrl-brl:>+7.4f}")

# ============================================================
# STEP 6: GENERATE TEST SUBMISSION (only if improved)
# ============================================================
log(f"\n{'='*60}")
log("STEP 6: Generate test submission")
log(f"{'='*60}")

if n_r1 + n_rl > b_r1 + b_rl:
    log("ROUGE-aligned model is BETTER! Generating submission...")
    use_index = fidx_new
    use_model_name = "rouge_aligned"

    log("Encoding test...")
    test_emb = bienc.encode(
        [f"{PREFIX}{q}" for q in test_df['input'].fillna('').astype(str).tolist()],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
else:
    log("ROUGE-aligned model is NOT better. Using baseline for submission.")
    use_index = faiss.IndexFlatIP(corpus_emb.shape[1])
    use_index.add(corpus_emb)
    use_model_name = "baseline"

    test_emb = bienc.encode(
        [f"{PREFIX}{q}" for q in test_df['input'].fillna('').astype(str).tolist()],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

test_qs = test_df['input'].fillna('').astype(str).tolist()
rows = []
for i in tqdm(range(len(test_df)), desc="Test submission"):
    q = test_qs[i].strip()
    D, I = use_index.search(test_emb[i:i+1], 5)
    answer = "No answer found."
    for j in range(5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        answer = str(combined.iloc[ci]['output'])
        break

    rows.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
    })

sub = pd.DataFrame(rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub) == len(sample_sub)
fname = f'submission_{use_model_name}.csv'
sub.to_csv(OUTPUT_DIR / fname, index=False)
log(f"Saved: {fname}")

# ============================================================
# STEP 7: TRY SECOND EPOCH (if first helped)
# ============================================================
if n_r1 + n_rl > b_r1 + b_rl:
    log(f"\n{'='*60}")
    log("STEP 7: Training epoch 2 (pushing further)")
    log(f"{'='*60}")

    epoch1_r1, epoch1_rl = n_r1, n_rl

    # Rebuild ROUGE-aligned data with the improved model
    log("Rebuilding ROUGE-aligned data with improved embeddings...")
    train_q_emb2 = bienc.encode(
        [f"{PREFIX}{q}" for q in train_qs_text],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    rouge_pairs_2 = []
    for i in tqdm(range(len(train_df)), desc="Epoch 2 pairs"):
        q = train_qs_text[i].strip()
        ref = train_as_text[i].strip()
        if not q or not ref: continue

        D, I = fidx_new.search(train_q_emb2[i:i+1], CAND_K + 5)
        candidates = []
        for j in range(CAND_K + 5):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            cq = str(combined.iloc[ci]['input']).strip()
            if cq == q: continue
            ca = str(combined.iloc[ci]['output']).strip()
            candidates.append((ci, cq, ca))
            if len(candidates) >= CAND_K: break

        if len(candidates) < 3: continue

        rouge_scores = [fast_rouge1_f1(ref, ca) for _, _, ca in candidates]
        best_idx = int(np.argmax(rouge_scores))

        if best_idx > 0 and rouge_scores[best_idx] - rouge_scores[0] > 0.05:
            rouge_pairs_2.append(InputExample(
                texts=[f"{PREFIX}{q}", f"{PREFIX}{candidates[best_idx][1]}"]
            ))
        if rouge_scores[0] > 0.3:
            rouge_pairs_2.append(InputExample(
                texts=[f"{PREFIX}{q}", f"{PREFIX}{candidates[0][1]}"]
            ))

    del train_q_emb2; gc.collect()
    log(f"Epoch 2 pairs: {len(rouge_pairs_2)}")

    loader_2 = DataLoader(rouge_pairs_2, batch_size=16, shuffle=True)
    bienc.fit(
        train_objectives=[(loader_2, train_loss)],
        epochs=1,
        warmup_steps=int(len(loader_2) * 0.1),
        optimizer_params={'lr': 5e-6},  # even lower LR
        show_progress_bar=True,
        output_path=str(OUTPUT_DIR / 'afrie5-rouge-aligned-v2'),
    )
    bienc.save(str(OUTPUT_DIR / 'afrie5-rouge-aligned-v2'))
    log("Epoch 2 complete! Model saved.")

    # Evaluate epoch 2
    log("Evaluating epoch 2...")
    corpus_emb_v2 = bienc.encode(
        [f"{PREFIX}{q}" for q in questions_raw],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    fidx_v2 = faiss.IndexFlatIP(corpus_emb_v2.shape[1])
    fidx_v2.add(corpus_emb_v2)

    val_emb_v2 = bienc.encode(
        [f"{PREFIX}{q}" for q in val_qs],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    v2_r1s, v2_rls = [], []
    for i in tqdm(range(len(val_df)), desc="Epoch 2 eval"):
        q = str(val_df.iloc[i]['input']).strip()
        ref = str(val_df.iloc[i]['output']).strip()
        if not ref: continue
        D, I = fidx_v2.search(val_emb_v2[i:i+1], 5)
        answer = ''
        for j in range(5):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q: continue
            answer = str(combined.iloc[ci]['output'])
            break
        if not answer: answer = str(combined.iloc[int(I[0][0])]['output'])
        r = scorer.score(ref, answer)
        v2_r1s.append(r['rouge1'].fmeasure)
        v2_rls.append(r['rougeL'].fmeasure)

    v2_r1, v2_rl = np.mean(v2_r1s), np.mean(v2_rls)

    log(f"\n{'Method':<30} {'ROUGE-1':>10} {'ROUGE-L':>10}")
    log(f"{'-'*52}")
    log(f"{'Baseline':30} {b_r1:>10.4f} {b_rl:>10.4f}")
    log(f"{'Epoch 1':30} {epoch1_r1:>10.4f} {epoch1_rl:>10.4f}")
    log(f"{'Epoch 2':30} {v2_r1:>10.4f} {v2_rl:>10.4f}")

    # Generate epoch 2 submission if better
    if v2_r1 + v2_rl > epoch1_r1 + epoch1_rl:
        log("\nEpoch 2 is better! Generating submission...")
        test_emb_v2 = bienc.encode(
            [f"{PREFIX}{q}" for q in test_df['input'].fillna('').astype(str).tolist()],
            batch_size=64, show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)

        rows_v2 = []
        for i in tqdm(range(len(test_df)), desc="Epoch 2 submission"):
            q = test_qs[i].strip()
            D, I = fidx_v2.search(test_emb_v2[i:i+1], 5)
            answer = "No answer found."
            for j in range(5):
                ci = int(I[0][j])
                if ci >= len(combined): continue
                if str(combined.iloc[ci]['input']).strip() == q: continue
                answer = str(combined.iloc[ci]['output'])
                break
            rows_v2.append({
                'ID': test_df.iloc[i]['ID'],
                'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
            })

        sub_v2 = pd.DataFrame(rows_v2)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
        assert len(sub_v2) == len(sample_sub)
        sub_v2.to_csv(OUTPUT_DIR / 'submission_rouge_aligned_v2.csv', index=False)
        log("Saved: submission_rouge_aligned_v2.csv")
    else:
        log("Epoch 2 did NOT improve. Stick with epoch 1 submission.")

else:
    log("\nSkipping epoch 2 (epoch 1 didn't help).")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*60}")
log("DONE")
log(f"{'='*60}")
log(f"Previous LB best: 0.6545")
log(f"Submissions saved to Drive:")
for f in sorted(OUTPUT_DIR.glob("submission_*.csv")):
    log(f"  → {f.name}")
