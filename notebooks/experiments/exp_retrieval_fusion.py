"""
=============================================================================
RETRIEVAL FUSION: BM25 + AfriE5 + Per-Column Optimization
=============================================================================
Why this works:
- ROUGE measures WORD OVERLAP. BM25 directly measures word overlap.
- AfriE5 measures MEANING. Together they find answers that are
  both semantically right AND have high word overlap.
- Per-column: different answers for ROUGE-1, ROUGE-L, and LLM-Judge.

No generation. No guessing. Pure retrieval optimization.

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm rank-bm25 scikit-learn
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
from rank_bm25 import BM25Okapi

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

SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Ghana)', 'Amh_Eth': 'Amharic (Ethiopia)',
    'Eng_Eth': 'English (Ethiopia)', 'Eng_Gha': 'English (Ghana)',
    'Eng_Ken': 'English (Kenya)', 'Eng_Uga': 'English (Uganda)',
    'Lug_Uga': 'Luganda (Uganda)', 'Swa_Ken': 'Swahili (Kenya)',
}

if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# INDEX 1: AfriE5 Question-to-Question (current approach)
# ============================================================
log(f"\n{'='*60}")
log("INDEX 1: AfriE5 Q→Q (semantic)")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer

PREFIX_Q = "query: "
PREFIX_A = "passage: "

if AFRIE5_DIR and AFRIE5_DIR.exists():
    bienc = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0')
    log(f"AfriE5 from Drive: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")
else:
    bienc = SentenceTransformer('McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
    log(f"AfriE5 from HuggingFace: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

# Encode all corpus questions
log("Encoding corpus questions (Q→Q index)...")
corpus_q_emb = bienc.encode(
    [f"{PREFIX_Q}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
qq_index = faiss.IndexFlatIP(corpus_q_emb.shape[1])
qq_index.add(corpus_q_emb)
log(f"  Q→Q index: {corpus_q_emb.shape}")

# ============================================================
# INDEX 2: AfriE5 Question-to-Answer (new signal!)
# ============================================================
log(f"\n{'='*60}")
log("INDEX 2: AfriE5 Q→A (semantic, answer-level)")
log(f"{'='*60}")

log("Encoding corpus answers with 'passage:' prefix...")
corpus_a_emb = bienc.encode(
    [f"{PREFIX_A}{a}" for a in answers_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
qa_index = faiss.IndexFlatIP(corpus_a_emb.shape[1])
qa_index.add(corpus_a_emb)
log(f"  Q→A index: {corpus_a_emb.shape}")

# ============================================================
# INDEX 3 & 4: BM25 (lexical — directly measures word overlap)
# ============================================================
log(f"\n{'='*60}")
log("INDEX 3 & 4: BM25 (lexical matching)")
log(f"{'='*60}")

def tokenize_text(text):
    """Simple multilingual tokenizer: lowercase + split on non-alphanumeric."""
    text = str(text).lower().strip()
    tokens = re.findall(r'\w+', text, re.UNICODE)
    return tokens if tokens else ['']

log("Building BM25 index on questions...")
corpus_q_tokens = [tokenize_text(q) for q in tqdm(questions_raw, desc="Tokenize Q")]
bm25_qq = BM25Okapi(corpus_q_tokens)
log(f"  BM25 Q→Q: {len(corpus_q_tokens)} docs")

log("Building BM25 index on answers...")
corpus_a_tokens = [tokenize_text(a) for a in tqdm(answers_raw, desc="Tokenize A")]
bm25_qa = BM25Okapi(corpus_a_tokens)
log(f"  BM25 Q→A: {len(corpus_a_tokens)} docs")

# ============================================================
# ENCODE VAL AND TEST QUERIES
# ============================================================
log(f"\n{'='*60}")
log("Encoding val & test queries")
log(f"{'='*60}")

val_qs = val_df['input'].fillna('').astype(str).tolist()
val_q_emb = bienc.encode(
    [f"{PREFIX_Q}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

test_qs = test_df['input'].fillna('').astype(str).tolist()
test_q_emb = bienc.encode(
    [f"{PREFIX_Q}{q}" for q in test_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

bienc.cpu(); gc.collect(); torch.cuda.empty_cache()
log("All embeddings done. GPU freed.")

# ============================================================
# RETRIEVAL FUSION FUNCTIONS
# ============================================================

def get_candidates_multi(query_text, query_emb, top_k=50):
    """Get candidate indices from all 4 retrieval methods."""
    q_tokens = tokenize_text(query_text)
    query_stripped = query_text.strip()

    rankings = {}

    # 1. AfriE5 Q→Q
    D_qq, I_qq = qq_index.search(query_emb.reshape(1, -1), top_k + 5)
    qq_ranking = []
    for j in range(top_k + 5):
        ci = int(I_qq[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == query_stripped: continue
        qq_ranking.append(ci)
        if len(qq_ranking) >= top_k: break
    rankings['afrie5_qq'] = qq_ranking

    # 2. AfriE5 Q→A
    D_qa, I_qa = qa_index.search(query_emb.reshape(1, -1), top_k + 5)
    qa_ranking = []
    for j in range(top_k + 5):
        ci = int(I_qa[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == query_stripped: continue
        qa_ranking.append(ci)
        if len(qa_ranking) >= top_k: break
    rankings['afrie5_qa'] = qa_ranking

    # 3. BM25 Q→Q
    bm25_qq_scores = bm25_qq.get_scores(q_tokens)
    bm25_qq_top = np.argsort(bm25_qq_scores)[::-1]
    bm25_qq_ranking = []
    for ci in bm25_qq_top:
        ci = int(ci)
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == query_stripped: continue
        bm25_qq_ranking.append(ci)
        if len(bm25_qq_ranking) >= top_k: break
    rankings['bm25_qq'] = bm25_qq_ranking

    # 4. BM25 Q→A
    bm25_qa_scores = bm25_qa.get_scores(q_tokens)
    bm25_qa_top = np.argsort(bm25_qa_scores)[::-1]
    bm25_qa_ranking = []
    for ci in bm25_qa_top:
        ci = int(ci)
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == query_stripped: continue
        bm25_qa_ranking.append(ci)
        if len(bm25_qa_ranking) >= top_k: break
    rankings['bm25_qa'] = bm25_qa_ranking

    return rankings


def rrf_fuse(rankings_dict, method_weights=None, k=60):
    """Reciprocal Rank Fusion with optional per-method weights."""
    if method_weights is None:
        method_weights = {m: 1.0 for m in rankings_dict}

    scores = defaultdict(float)
    for method, ranking in rankings_dict.items():
        w = method_weights.get(method, 1.0)
        for rank, idx in enumerate(ranking):
            scores[idx] += w / (k + rank + 1)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ============================================================
# EVALUATE ON VALIDATION SET
# ============================================================
log(f"\n{'='*60}")
log("EVALUATING ALL STRATEGIES ON VALIDATION")
log(f"{'='*60}")

# Test multiple configurations
configs = {
    'afrie5_qq_only': {'afrie5_qq': 1.0, 'afrie5_qa': 0.0, 'bm25_qq': 0.0, 'bm25_qa': 0.0},
    'afrie5_qa_only': {'afrie5_qq': 0.0, 'afrie5_qa': 1.0, 'bm25_qq': 0.0, 'bm25_qa': 0.0},
    'bm25_qq_only':   {'afrie5_qq': 0.0, 'afrie5_qa': 0.0, 'bm25_qq': 1.0, 'bm25_qa': 0.0},
    'bm25_qa_only':   {'afrie5_qq': 0.0, 'afrie5_qa': 0.0, 'bm25_qq': 0.0, 'bm25_qa': 1.0},
    'dense_both':     {'afrie5_qq': 1.0, 'afrie5_qa': 1.0, 'bm25_qq': 0.0, 'bm25_qa': 0.0},
    'bm25_both':      {'afrie5_qq': 0.0, 'afrie5_qa': 0.0, 'bm25_qq': 1.0, 'bm25_qa': 1.0},
    'all_equal':      {'afrie5_qq': 1.0, 'afrie5_qa': 1.0, 'bm25_qq': 1.0, 'bm25_qa': 1.0},
    'dense_heavy':    {'afrie5_qq': 2.0, 'afrie5_qa': 1.0, 'bm25_qq': 0.5, 'bm25_qa': 0.5},
    'bm25_heavy':     {'afrie5_qq': 0.5, 'afrie5_qa': 0.5, 'bm25_qq': 2.0, 'bm25_qa': 1.0},
    'qq_heavy':       {'afrie5_qq': 2.0, 'afrie5_qa': 0.5, 'bm25_qq': 2.0, 'bm25_qa': 0.5},
    'qa_heavy':       {'afrie5_qq': 0.5, 'afrie5_qa': 2.0, 'bm25_qq': 0.5, 'bm25_qa': 2.0},
}

# Gather all candidate sets for val
log("Retrieving candidates for all val questions...")
val_candidates = []
for i in tqdm(range(len(val_df)), desc="Val retrieval"):
    cands = get_candidates_multi(val_qs[i], val_q_emb[i], top_k=50)
    val_candidates.append(cands)

# Evaluate each config
log("\nEvaluating fusion configs...")
results = {}
per_lang_results = {}

for config_name, weights in tqdm(configs.items(), desc="Configs"):
    r1s, rls = [], []
    lang_scores = defaultdict(lambda: {'r1': [], 'rl': []})

    for i in range(len(val_df)):
        ref = str(val_df.iloc[i]['output']).strip()
        sub = str(val_df.iloc[i]['subset'])
        if not ref: continue

        # Only include methods with weight > 0
        active = {m: r for m, r in val_candidates[i].items() if weights.get(m, 0) > 0 and r}
        if not active:
            r1s.append(0); rls.append(0); continue

        fused = rrf_fuse(active, weights)
        if not fused:
            r1s.append(0); rls.append(0); continue

        answer = str(combined.iloc[fused[0]]['output'])
        r = scorer.score(ref, answer)
        r1s.append(r['rouge1'].fmeasure)
        rls.append(r['rougeL'].fmeasure)
        lang_scores[sub]['r1'].append(r['rouge1'].fmeasure)
        lang_scores[sub]['rl'].append(r['rougeL'].fmeasure)

    results[config_name] = (np.mean(r1s), np.mean(rls))
    per_lang_results[config_name] = {
        sub: (np.mean(d['r1']), np.mean(d['rl']))
        for sub, d in lang_scores.items()
    }

# Print results sorted by combined score
log(f"\n{'Config':<20} {'ROUGE-1':>10} {'ROUGE-L':>10} {'Combined':>10}")
log(f"{'-'*52}")
sorted_configs = sorted(results.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)
for name, (r1, rl) in sorted_configs:
    marker = " ★" if name == sorted_configs[0][0] else ""
    log(f"{name:<20} {r1:>10.4f} {rl:>10.4f} {r1+rl:>10.4f}{marker}")

best_config_name = sorted_configs[0][0]
best_weights = configs[best_config_name]
log(f"\nBest config: {best_config_name} -> weights: {best_weights}")

# Per-language breakdown for best config
log(f"\nPer-language for '{best_config_name}':")
log(f"{'Subset':<12} {'R1':>8} {'RL':>8}")
best_lang = per_lang_results[best_config_name]
for sub in sorted(best_lang.keys()):
    r1, rl = best_lang[sub]
    log(f"  {sub:<12} {r1:>8.4f} {rl:>8.4f}")

# ============================================================
# PER-COLUMN OPTIMIZATION
# ============================================================
log(f"\n{'='*60}")
log("PER-COLUMN OPTIMIZATION")
log(f"{'='*60}")
log("Finding best config for each metric separately...")

# Find best config for ROUGE-1
best_r1_config = max(results.items(), key=lambda x: x[1][0])
# Find best config for ROUGE-L
best_rl_config = max(results.items(), key=lambda x: x[1][1])

log(f"Best for ROUGE-1: {best_r1_config[0]} (R1={best_r1_config[1][0]:.4f})")
log(f"Best for ROUGE-L: {best_rl_config[0]} (RL={best_rl_config[1][1]:.4f})")

# For LLM-Judge, semantic quality matters most — try AfriE5 Q→Q
log(f"For LLM-Judge: using {best_config_name} (same as best combined)")

r1_weights = configs[best_r1_config[0]]
rl_weights = configs[best_rl_config[0]]
llm_weights = best_weights

# ============================================================
# ORACLE ANALYSIS (best possible from fusion candidates)
# ============================================================
log(f"\n{'='*60}")
log("ORACLE: What's the ceiling with 4-way fusion?")
log(f"{'='*60}")

oracle_r1s, oracle_rls = [], []
for i in tqdm(range(len(val_df)), desc="Oracle"):
    ref = str(val_df.iloc[i]['output']).strip()
    if not ref: continue

    # Collect ALL unique candidates from all 4 methods
    all_cands = set()
    for method_cands in val_candidates[i].values():
        all_cands.update(method_cands[:50])

    best_r1, best_rl = 0, 0
    for ci in all_cands:
        answer = str(combined.iloc[ci]['output'])
        r = scorer.score(ref, answer)
        s_r1 = r['rouge1'].fmeasure
        s_rl = r['rougeL'].fmeasure
        if s_r1 + s_rl > best_r1 + best_rl:
            best_r1, best_rl = s_r1, s_rl

    oracle_r1s.append(best_r1)
    oracle_rls.append(best_rl)

log(f"Oracle (best from all 4 indices): R1={np.mean(oracle_r1s):.4f} RL={np.mean(oracle_rls):.4f}")
log(f"Best fusion result:               R1={sorted_configs[0][1][0]:.4f} RL={sorted_configs[0][1][1]:.4f}")
log(f"Remaining gap:                     R1={np.mean(oracle_r1s)-sorted_configs[0][1][0]:+.4f} RL={np.mean(oracle_rls)-sorted_configs[0][1][1]:+.4f}")

# ============================================================
# GENERATE TEST SUBMISSIONS
# ============================================================
log(f"\n{'='*60}")
log("GENERATING TEST SUBMISSIONS")
log(f"{'='*60}")

# Submission 1: Best overall config (same answer in all 3 columns)
log(f"\nSubmission 1: Best config '{best_config_name}'")
rows_best = []
for i in tqdm(range(len(test_df)), desc="Best config"):
    q = test_qs[i]
    cands = get_candidates_multi(q, test_q_emb[i], top_k=50)
    active = {m: r for m, r in cands.items() if best_weights.get(m, 0) > 0 and r}
    if active:
        fused = rrf_fuse(active, best_weights)
        answer = str(combined.iloc[fused[0]]['output'])
    else:
        answer = "No answer found."
    rows_best.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
    })

sub_best = pd.DataFrame(rows_best)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_best) == len(sample_sub)
sub_best.to_csv(OUTPUT_DIR / 'submission_fusion_best.csv', index=False)
log(f"Saved: submission_fusion_best.csv")

# Submission 2: Per-column optimized (different answer per metric)
log(f"\nSubmission 2: Per-column optimized")
rows_percol = []
for i in tqdm(range(len(test_df)), desc="Per-column"):
    q = test_qs[i]
    cands = get_candidates_multi(q, test_q_emb[i], top_k=50)

    # ROUGE-1 optimized answer
    active_r1 = {m: r for m, r in cands.items() if r1_weights.get(m, 0) > 0 and r}
    if active_r1:
        fused_r1 = rrf_fuse(active_r1, r1_weights)
        ans_r1 = str(combined.iloc[fused_r1[0]]['output'])
    else:
        ans_r1 = "No answer found."

    # ROUGE-L optimized answer
    active_rl = {m: r for m, r in cands.items() if rl_weights.get(m, 0) > 0 and r}
    if active_rl:
        fused_rl = rrf_fuse(active_rl, rl_weights)
        ans_rl = str(combined.iloc[fused_rl[0]]['output'])
    else:
        ans_rl = "No answer found."

    # LLM optimized answer
    active_llm = {m: r for m, r in cands.items() if llm_weights.get(m, 0) > 0 and r}
    if active_llm:
        fused_llm = rrf_fuse(active_llm, llm_weights)
        ans_llm = str(combined.iloc[fused_llm[0]]['output'])
    else:
        ans_llm = "No answer found."

    rows_percol.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': ans_r1,
        'TargetRLF1': ans_rl,
        'TargetLLM': ans_llm,
    })

sub_percol = pd.DataFrame(rows_percol)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_percol) == len(sample_sub)
sub_percol.to_csv(OUTPUT_DIR / 'submission_fusion_percol.csv', index=False)
log(f"Saved: submission_fusion_percol.csv")

# Submission 3: AfriE5 Q→Q top-1 baseline (for comparison)
log(f"\nSubmission 3: AfriE5 Q→Q baseline")
rows_base = []
for i in tqdm(range(len(test_df)), desc="Baseline"):
    q = test_qs[i]
    D, I = qq_index.search(test_q_emb[i:i+1], 5)
    answer = "No answer found."
    for j in range(5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q.strip(): continue
        answer = str(combined.iloc[ci]['output'])
        break
    rows_base.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
    })

sub_base = pd.DataFrame(rows_base)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_base) == len(sample_sub)
sub_base.to_csv(OUTPUT_DIR / 'submission_baseline_qq.csv', index=False)
log(f"Saved: submission_baseline_qq.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*60}")
log("FINAL SUMMARY")
log(f"{'='*60}")

log(f"\n{'Config':<20} {'ROUGE-1':>10} {'ROUGE-L':>10} {'Combined':>10}")
log(f"{'-'*52}")
for name, (r1, rl) in sorted_configs[:5]:
    log(f"{name:<20} {r1:>10.4f} {rl:>10.4f} {r1+rl:>10.4f}")
log(f"{'-'*52}")
log(f"{'Oracle ceiling':<20} {np.mean(oracle_r1s):>10.4f} {np.mean(oracle_rls):>10.4f}")

log(f"\nPrevious best LB: 0.6545")
log(f"\n3 submissions saved to Drive:")
log(f"  1. submission_fusion_best.csv    — best fusion config")
log(f"  2. submission_fusion_percol.csv  — per-column optimized")
log(f"  3. submission_baseline_qq.csv    — AfriE5 Q→Q baseline")
log(f"\nSubmit fusion_best first. If it beats 0.6545, try percol next.")
log("Done!")
