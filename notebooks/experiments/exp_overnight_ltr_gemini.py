"""
=============================================================================
OVERNIGHT MEGA-PIPELINE: LTR Reranker + Gemini Extractive + LLM Quality
=============================================================================
Three independent approaches, each saves its own submission.

Phase 0 (~15 min): Load data, model, build all indices + BM25
Phase 1 (~40 min): LTR feature extraction + XGBoost training
Phase 2 (~30 min): Gemini val test (200 samples)
Phase 3 (~10 min): LTR val evaluation + test submission
Phase 4 (~4 hr):   Gemini test generation (ROUGE + LLM)
Phase 5 (~5 min):  Create all submission variants

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2:
    import os
    os.environ['GEMINI_API_KEY'] = 'YOUR_KEY_HERE'
Cell 3:
    !pip install -q sentence-transformers faiss-cpu rouge-score tqdm rank-bm25 xgboost google-genai
Cell 4: Paste this entire script
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
import json
import time
import traceback
import pickle
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime
from collections import defaultdict
from rank_bm25 import BM25Okapi
import xgboost as xgb

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# PATHS + DATA
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

log("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df   = pd.read_csv(DATA_DIR / 'Val.csv')
test_df  = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])
combined = combined.reset_index(drop=True)
questions_raw = combined['input'].fillna('').astype(str).tolist()
answers_raw   = combined['output'].fillna('').astype(str).tolist()
subsets_raw   = combined['subset'].fillna('').astype(str).tolist()

log(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Combined: {len(combined)}")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Ghana)', 'Amh_Eth': 'Amharic (Ethiopia)',
    'Eng_Eth': 'English (Ethiopia)', 'Eng_Gha': 'English (Ghana)',
    'Eng_Ken': 'English (Kenya)', 'Eng_Uga': 'English (Uganda)',
    'Lug_Uga': 'Luganda (Uganda)', 'Swa_Ken': 'Swahili (Kenya)',
}

if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def tokenize_text(text):
    return re.findall(r'\w+', str(text).lower(), re.UNICODE) or ['']

def fast_rouge1_f1(ref, hyp):
    ref_t = set(str(ref).lower().split())
    hyp_t = set(str(hyp).lower().split())
    if not ref_t or not hyp_t: return 0.0
    ov = len(ref_t & hyp_t)
    if ov == 0: return 0.0
    p, r = ov / len(hyp_t), ov / len(ref_t)
    return 2 * p * r / (p + r)

# ============================================================
# PHASE 0: LOAD MODEL + BUILD ALL INDICES
# ============================================================
log(f"\n{'='*60}")
log("PHASE 0: Load model + Build indices")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer
PREFIX = "query: "
PREFIX_A = "passage: "

bienc = SentenceTransformer(str(AFRIE5_DIR) if AFRIE5_DIR and AFRIE5_DIR.exists()
    else 'McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
log(f"AfriE5: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

# Encode corpus questions
log("Encoding corpus questions...")
corpus_q_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

# Encode corpus answers (for Q->A similarity feature)
log("Encoding corpus answers...")
corpus_a_emb = bienc.encode(
    [f"{PREFIX_A}{a}" for a in answers_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

# FAISS indices
global_qq_idx = faiss.IndexFlatIP(corpus_q_emb.shape[1])
global_qq_idx.add(corpus_q_emb)

global_qa_idx = faiss.IndexFlatIP(corpus_a_emb.shape[1])
global_qa_idx.add(corpus_a_emb)

# Per-language indices
lang_indices = {}
for sub in sorted(set(subsets_raw)):
    mask = [i for i, s in enumerate(subsets_raw) if s == sub]
    sub_emb = corpus_q_emb[mask]
    idx = faiss.IndexFlatIP(sub_emb.shape[1])
    idx.add(sub_emb)
    lang_indices[sub] = (idx, mask)

# BM25 indices
log("Building BM25 indices...")
corpus_q_tokens = [tokenize_text(q) for q in questions_raw]
corpus_a_tokens = [tokenize_text(a) for a in answers_raw]
bm25_qq = BM25Okapi(corpus_q_tokens)
bm25_qa = BM25Okapi(corpus_a_tokens)

# Encode val + test + train queries
log("Encoding val queries...")
val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

log("Encoding test queries...")
test_qs = test_df['input'].fillna('').astype(str).tolist()
test_subs = test_df['subset'].fillna('').astype(str).tolist()
test_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in test_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

log("Encoding train queries...")
train_qs = train_df['input'].fillna('').astype(str).tolist()
train_as = train_df['output'].fillna('').astype(str).tolist()
train_subs = train_df['subset'].fillna('').astype(str).tolist()
train_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in train_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

bienc.cpu(); gc.collect(); torch.cuda.empty_cache()
log("All embeddings done. GPU freed.")

# ============================================================
# PHASE 1: LTR FEATURE EXTRACTION + XGBOOST TRAINING
# ============================================================
log(f"\n{'='*60}")
log("PHASE 1: Learning-to-Rank — Feature extraction + training")
log(f"{'='*60}")

CAND_K = 30  # candidates per query

def extract_features(q_text, q_emb, candidate_idx, qq_sim):
    """Extract features for a (query, candidate) pair."""
    q_stripped = q_text.strip()
    ci = candidate_idx
    cq = str(combined.iloc[ci]['input']).strip()
    ca = str(combined.iloc[ci]['output']).strip()
    c_sub = str(combined.iloc[ci]['subset'])

    q_tokens = tokenize_text(q_stripped)
    a_tokens = tokenize_text(ca)
    cq_tokens = tokenize_text(cq)

    q_words = set(q_tokens)
    a_words = set(a_tokens)
    cq_words = set(cq_tokens)

    # Q->A cosine similarity
    qa_sim = float(np.dot(q_emb, corpus_a_emb[ci]))

    # BM25 scores
    bm25_qq_score = float(bm25_qq.get_scores(q_tokens)[ci])
    bm25_qa_score = float(bm25_qa.get_scores(q_tokens)[ci])

    # Word overlaps
    qa_overlap = len(q_words & a_words) / max(len(q_words), 1)
    qq_overlap = len(q_words & cq_words) / max(len(q_words), 1)

    features = [
        qq_sim,                          # 0: AfriE5 Q-Q cosine sim
        qa_sim,                          # 1: AfriE5 Q-A cosine sim
        bm25_qq_score,                   # 2: BM25 Q-Q
        bm25_qa_score,                   # 3: BM25 Q-A
        len(a_tokens),                   # 4: answer length (words)
        len(q_tokens),                   # 5: question length (words)
        len(a_tokens) / max(len(q_tokens), 1),  # 6: A/Q length ratio
        qa_overlap,                      # 7: Q-A word overlap fraction
        qq_overlap,                      # 8: Q-Q word overlap fraction
        len(a_words),                    # 9: unique words in answer
        1.0 if c_sub == q_stripped else 0.0,  # dummy, replaced below
    ]
    return features

def extract_features_batch(q_text, q_emb, q_subset, candidates_with_sims):
    """Extract features for all candidates of a query."""
    q_tokens = tokenize_text(q_text)
    q_words = set(q_tokens)
    q_len = len(q_tokens)

    # Batch BM25 (computed once per query)
    bm25_qq_all = bm25_qq.get_scores(q_tokens)
    bm25_qa_all = bm25_qa.get_scores(q_tokens)

    features_list = []
    for ci, qq_sim in candidates_with_sims:
        ca = str(combined.iloc[ci]['output']).strip()
        cq = str(combined.iloc[ci]['input']).strip()
        c_sub = str(combined.iloc[ci]['subset'])

        a_tokens = tokenize_text(ca)
        cq_tokens = tokenize_text(cq)
        a_words = set(a_tokens)
        cq_words = set(cq_tokens)

        qa_sim = float(np.dot(q_emb, corpus_a_emb[ci]))
        a_len = len(a_tokens)

        features = [
            float(qq_sim),                        # 0
            qa_sim,                               # 1
            float(bm25_qq_all[ci]),                # 2
            float(bm25_qa_all[ci]),                # 3
            a_len,                                 # 4
            q_len,                                 # 5
            a_len / max(q_len, 1),                # 6
            len(q_words & a_words) / max(len(q_words), 1),  # 7
            len(q_words & cq_words) / max(len(q_words), 1), # 8
            len(a_words),                          # 9
            1.0 if c_sub == q_subset else 0.0,     # 10: language match
            abs(a_len - q_len),                    # 11: length difference
        ]
        features_list.append(features)

    return features_list

# Build training data for LTR
log("Building LTR training data from train set...")
X_train_ltr, y_train_ltr, qids_train = [], [], []

for i in tqdm(range(len(train_df)), desc="LTR train features"):
    q = train_qs[i].strip()
    ref = train_as[i].strip()
    q_sub = train_subs[i]
    if not q or not ref: continue

    D, I = global_qq_idx.search(train_emb[i:i+1], CAND_K + 5)
    candidates = []
    for j in range(CAND_K + 5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        candidates.append((ci, float(D[0][j])))
        if len(candidates) >= CAND_K: break

    if len(candidates) < 5: continue

    feats = extract_features_batch(q, train_emb[i], q_sub, candidates)

    for feat_idx, (ci, _) in enumerate(candidates):
        ca = str(combined.iloc[ci]['output']).strip()
        rouge = fast_rouge1_f1(ref, ca)
        X_train_ltr.append(feats[feat_idx])
        y_train_ltr.append(rouge)
        qids_train.append(i)

X_train_ltr = np.array(X_train_ltr, dtype=np.float32)
y_train_ltr = np.array(y_train_ltr, dtype=np.float32)

# Group sizes for ranking
unique_qids = []
current_qid = -1
groups_train = []
count = 0
for qid in qids_train:
    if qid != current_qid:
        if count > 0:
            groups_train.append(count)
        current_qid = qid
        count = 1
    else:
        count += 1
if count > 0:
    groups_train.append(count)

log(f"LTR training data: {X_train_ltr.shape[0]} pairs, {len(groups_train)} queries")
log(f"Feature dim: {X_train_ltr.shape[1]}")
log(f"Label range: [{y_train_ltr.min():.3f}, {y_train_ltr.max():.3f}]")

# Build val data for LTR
log("Building LTR val data...")
X_val_ltr, y_val_ltr, qids_val = [], [], []
val_candidates_all = []  # Store for later use

for i in tqdm(range(len(val_df)), desc="LTR val features"):
    q = val_qs[i].strip()
    ref = str(val_df.iloc[i]['output']).strip()
    q_sub = str(val_df.iloc[i]['subset'])
    if not q or not ref:
        val_candidates_all.append([])  # placeholder to keep indices aligned
        continue

    D, I = global_qq_idx.search(val_emb[i:i+1], CAND_K + 5)
    candidates = []
    for j in range(CAND_K + 5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        candidates.append((ci, float(D[0][j])))
        if len(candidates) >= CAND_K: break

    val_candidates_all.append(candidates)  # always append at index i
    if len(candidates) < 3: continue

    feats = extract_features_batch(q, val_emb[i], q_sub, candidates)
    for feat_idx, (ci, _) in enumerate(candidates):
        ca = str(combined.iloc[ci]['output']).strip()
        rouge = fast_rouge1_f1(ref, ca)
        X_val_ltr.append(feats[feat_idx])
        y_val_ltr.append(rouge)
        qids_val.append(i)

X_val_ltr = np.array(X_val_ltr, dtype=np.float32)
y_val_ltr = np.array(y_val_ltr, dtype=np.float32)

groups_val = []
current_qid = -1
count = 0
for qid in qids_val:
    if qid != current_qid:
        if count > 0: groups_val.append(count)
        current_qid = qid
        count = 1
    else: count += 1
if count > 0: groups_val.append(count)

log(f"LTR val data: {X_val_ltr.shape[0]} pairs, {len(groups_val)} queries")

# Train XGBoost LambdaMART
log("\nTraining XGBoost LambdaMART ranker...")
dtrain = xgb.DMatrix(X_train_ltr, label=y_train_ltr)
dtrain.set_group(groups_train)
dval = xgb.DMatrix(X_val_ltr, label=y_val_ltr)
dval.set_group(groups_val)

params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'seed': 42,
}

model_ltr = xgb.train(
    params, dtrain,
    num_boost_round=300,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=30,
    verbose_eval=50,
)

model_ltr.save_model(str(OUTPUT_DIR / 'ltr_xgb_model.json'))
log("LTR model saved!")

# Feature importance
feat_names = ['qq_sim', 'qa_sim', 'bm25_qq', 'bm25_qa', 'a_len', 'q_len',
              'a_q_ratio', 'qa_overlap', 'qq_overlap', 'a_unique', 'lang_match', 'len_diff']
importance = model_ltr.get_score(importance_type='gain')
log("\nFeature importance:")
for f in sorted(importance, key=importance.get, reverse=True):
    fname = feat_names[int(f[1:])] if f.startswith('f') and f[1:].isdigit() else f
    log(f"  {fname}: {importance[f]:.1f}")

# ============================================================
# PHASE 2: LTR EVALUATION ON VAL
# ============================================================
log(f"\n{'='*60}")
log("PHASE 2: LTR evaluation on val")
log(f"{'='*60}")

ltr_r1s, ltr_rls = [], []
baseline_r1s, baseline_rls = [], []
per_lang_ltr = defaultdict(lambda: {'r1': [], 'rl': []})
per_lang_base = defaultdict(lambda: {'r1': [], 'rl': []})

for i in tqdm(range(len(val_df)), desc="LTR val eval"):
    q = val_qs[i].strip()
    ref = str(val_df.iloc[i]['output']).strip()
    q_sub = str(val_df.iloc[i]['subset'])
    if not q or not ref or i >= len(val_candidates_all): continue

    candidates = val_candidates_all[i]
    if not candidates: continue

    # Baseline: top-1
    top1_answer = str(combined.iloc[candidates[0][0]]['output'])
    r_base = scorer.score(ref, top1_answer)
    baseline_r1s.append(r_base['rouge1'].fmeasure)
    baseline_rls.append(r_base['rougeL'].fmeasure)
    per_lang_base[q_sub]['r1'].append(r_base['rouge1'].fmeasure)
    per_lang_base[q_sub]['rl'].append(r_base['rougeL'].fmeasure)

    # LTR: predict scores, pick best
    feats = extract_features_batch(q, val_emb[i], q_sub, candidates)
    X_pred = np.array(feats, dtype=np.float32)
    dpred = xgb.DMatrix(X_pred)
    scores = model_ltr.predict(dpred)
    best_idx = int(np.argmax(scores))
    ltr_answer = str(combined.iloc[candidates[best_idx][0]]['output'])

    r_ltr = scorer.score(ref, ltr_answer)
    ltr_r1s.append(r_ltr['rouge1'].fmeasure)
    ltr_rls.append(r_ltr['rougeL'].fmeasure)
    per_lang_ltr[q_sub]['r1'].append(r_ltr['rouge1'].fmeasure)
    per_lang_ltr[q_sub]['rl'].append(r_ltr['rougeL'].fmeasure)

b_r1, b_rl = np.mean(baseline_r1s), np.mean(baseline_rls)
l_r1, l_rl = np.mean(ltr_r1s), np.mean(ltr_rls)

log(f"\n{'Method':<25} {'R1':>8} {'RL':>8}")
log(f"{'-'*43}")
log(f"{'Baseline (top-1)':25} {b_r1:>8.4f} {b_rl:>8.4f}")
log(f"{'LTR reranked':25} {l_r1:>8.4f} {l_rl:>8.4f}")
log(f"{'Improvement':25} {l_r1-b_r1:>+8.4f} {l_rl-b_rl:>+8.4f}")

log(f"\nPer-language LTR vs baseline:")
log(f"{'Subset':<12} {'Base R1':>8} {'LTR R1':>8} {'Δ':>7}")
for sub in sorted(set(list(per_lang_base.keys()) + list(per_lang_ltr.keys()))):
    br = np.mean(per_lang_base[sub]['r1']) if per_lang_base[sub]['r1'] else 0
    lr = np.mean(per_lang_ltr[sub]['r1']) if per_lang_ltr[sub]['r1'] else 0
    marker = " ★" if lr > br + 0.005 else ""
    log(f"  {sub:<12} {br:>8.4f} {lr:>8.4f} {lr-br:>+7.4f}{marker}")

USE_LTR = (l_r1 + l_rl) > (b_r1 + b_rl)
log(f"\nDecision: {'USE LTR' if USE_LTR else 'KEEP baseline'}")

# Per-language LTR decision
USE_LTR_PER_LANG = {}
for sub in sorted(per_lang_base.keys()):
    br = np.mean(per_lang_base[sub]['r1'])
    lr = np.mean(per_lang_ltr[sub]['r1']) if per_lang_ltr[sub]['r1'] else 0
    USE_LTR_PER_LANG[sub] = lr > br + 0.003
    if USE_LTR_PER_LANG[sub]:
        log(f"  ★ {sub}: Use LTR (+{lr-br:.4f})")

# ============================================================
# PHASE 3: LTR TEST SUBMISSION
# ============================================================
log(f"\n{'='*60}")
log("PHASE 3: LTR test submission")
log(f"{'='*60}")

rows_ltr = []
for i in tqdm(range(len(test_df)), desc="LTR test"):
    q = test_qs[i].strip()
    q_sub = test_subs[i]

    D, I = global_qq_idx.search(test_emb[i:i+1], CAND_K + 5)
    candidates = []
    for j in range(CAND_K + 5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        candidates.append((ci, float(D[0][j])))
        if len(candidates) >= CAND_K: break

    if not candidates:
        rows_ltr.append({'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': 'No answer', 'TargetRLF1': 'No answer', 'TargetLLM': 'No answer'})
        continue

    # Decide per-language whether to use LTR
    use_ltr = USE_LTR_PER_LANG.get(q_sub, USE_LTR)

    if use_ltr and len(candidates) >= 3:
        feats = extract_features_batch(q, test_emb[i], q_sub, candidates)
        X_pred = np.array(feats, dtype=np.float32)
        dpred = xgb.DMatrix(X_pred)
        scores = model_ltr.predict(dpred)
        best_idx = int(np.argmax(scores))
        answer = str(combined.iloc[candidates[best_idx][0]]['output'])
    else:
        answer = str(combined.iloc[candidates[0][0]]['output'])

    rows_ltr.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': answer, 'TargetRLF1': answer, 'TargetLLM': answer,
    })

sub_ltr = pd.DataFrame(rows_ltr)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_ltr) == len(sample_sub)
sub_ltr.to_csv(OUTPUT_DIR / 'submission_ltr.csv', index=False)
log("Saved: submission_ltr.csv")

# ============================================================
# PHASE 4: GEMINI API
# ============================================================
log(f"\n{'='*60}")
log("PHASE 4: Gemini API")
log(f"{'='*60}")

api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY') or userdata.get('GOOGLE_API_KEY')
    except Exception:
        pass

GEMINI_OK = False
if api_key:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        test_r = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Say OK',
            config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=5),
        )
        log(f"Gemini API test: {test_r.text.strip()}")
        GEMINI_OK = True
    except Exception as e:
        log(f"Gemini API failed: {e}")
else:
    log("No Gemini API key. Skipping Gemini phases.")
    log("Set GEMINI_API_KEY in env or Colab secrets to enable.")

if GEMINI_OK:
    call_count, call_start = 0, time.time()

    def gemini_call(prompt, temp=0.3, max_tok=600, retries=3):
        global call_count, call_start
        for attempt in range(retries):
            try:
                call_count += 1
                elapsed = time.time() - call_start
                if elapsed < 60 and call_count > 14:
                    time.sleep(61 - elapsed)
                    call_count, call_start = 0, time.time()
                resp = client.models.generate_content(
                    model='gemini-2.0-flash', contents=prompt,
                    config=genai.types.GenerateContentConfig(temperature=temp, max_output_tokens=max_tok),
                )
                return resp.text.strip()
            except Exception as e:
                if '429' in str(e).lower() or 'quota' in str(e).lower():
                    wait = min(30 * (attempt + 1), 120)
                    log(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    call_count, call_start = 0, time.time()
                elif attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        return None

    def get_top_k(q, q_emb, q_sub, k=5):
        D, I = global_qq_idx.search(q_emb.reshape(1, -1), k * 3)
        results = []
        for j in range(k * 3):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q.strip(): continue
            results.append({'answer': str(combined.iloc[ci]['output']),
                            'subset': str(combined.iloc[ci]['subset'])})
            if len(results) >= k: break
        return results

    # --- Val test (200 samples) ---
    log("\nGemini val test (200 samples)...")
    gem_r1s, gem_rls = [], []
    for i in tqdm(range(min(200, len(val_df))), desc="Gemini val"):
        q = val_qs[i]
        ref = str(val_df.iloc[i]['output']).strip()
        sub = str(val_df.iloc[i]['subset'])
        lang = SUBSET_TO_LANG.get(sub, sub)
        if not ref: continue

        contexts = get_top_k(q, val_emb[i], sub, k=5)
        ctx_str = "\n".join([f"Answer {k+1}: {c['answer']}" for k, c in enumerate(contexts)])

        prompt = f"""You are given a health question and {len(contexts)} reference answers. Create the BEST answer by selecting and combining the most relevant sentences from these references.

RULES:
1. Use ONLY words and phrases from the provided references — copy them EXACTLY
2. Do NOT paraphrase or add new information
3. Combine the most relevant parts from different answers
4. Answer in {lang} (same language as question)
5. Keep similar length to references
6. Start directly — no preamble

Question: {q}

{ctx_str}

Combined answer:"""

        gen = gemini_call(prompt, temp=0.1)
        if gen:
            r = scorer.score(ref, gen)
            gem_r1s.append(r['rouge1'].fmeasure)
            gem_rls.append(r['rougeL'].fmeasure)
        else:
            # Compute baseline on-the-fly (safer than indexing)
            if i < len(val_candidates_all) and val_candidates_all[i]:
                fb_ans = str(combined.iloc[val_candidates_all[i][0][0]]['output'])
                r_fb = scorer.score(ref, fb_ans)
                gem_r1s.append(r_fb['rouge1'].fmeasure)
                gem_rls.append(r_fb['rougeL'].fmeasure)
            else:
                gem_r1s.append(0)
                gem_rls.append(0)

        if (i+1) % 50 == 0:
            log(f"  [{i+1}] Gemini R1={np.mean(gem_r1s):.4f} Base R1={b_r1:.4f}")

    gem_r1, gem_rl = np.mean(gem_r1s), np.mean(gem_rls)
    log(f"\nGemini val: R1={gem_r1:.4f} RL={gem_rl:.4f}")
    log(f"Baseline:   R1={np.mean(baseline_r1s[:200]):.4f} RL={np.mean(baseline_rls[:200]):.4f}")
    USE_GEMINI_ROUGE = gem_r1 > np.mean(baseline_r1s[:200])

    # --- Test generation ---
    log("\nGenerating test answers with Gemini...")
    rouge_prog = OUTPUT_DIR / 'gemini_rouge_prog.json'
    llm_prog = OUTPUT_DIR / 'gemini_llm_prog.json'

    rouge_ans = json.load(open(rouge_prog)) if rouge_prog.exists() else {}
    llm_ans = json.load(open(llm_prog)) if llm_prog.exists() else {}
    log(f"Resume: {len(rouge_ans)} ROUGE, {len(llm_ans)} LLM done")

    for i in tqdm(range(len(test_df)), desc="Gemini test"):
        rid = str(test_df.iloc[i]['ID'])
        q = test_qs[i]
        sub = test_subs[i]
        lang = SUBSET_TO_LANG.get(sub, sub)

        contexts = get_top_k(q, test_emb[i], sub, k=5)
        ctx_str = "\n".join([f"Answer {k+1}: {c['answer']}" for k, c in enumerate(contexts)])
        fallback = contexts[0]['answer'] if contexts else "No answer."

        # ROUGE answer
        if rid not in rouge_ans:
            prompt_r = f"""You are given a health question and reference answers. Create the BEST answer by combining the most relevant sentences.

RULES: Use ONLY exact words/phrases from references. Do NOT paraphrase. Answer in {lang}. Start directly.

Question: {q}

{ctx_str}

Combined answer:"""
            gen_r = gemini_call(prompt_r, temp=0.1)
            rouge_ans[rid] = gen_r if gen_r else fallback

        # LLM answer
        if rid not in llm_ans:
            prompt_l = f"""You are a multilingual health expert. Answer accurately and comprehensively in {lang}.

Question: {q}

Reference information:
{ctx_str}

Provide a thorough, well-organized, culturally appropriate answer. Use accurate medical terminology. Answer directly:"""
            gen_l = gemini_call(prompt_l, temp=0.3)
            llm_ans[rid] = gen_l if gen_l else fallback

        if (i+1) % 100 == 0:
            json.dump(rouge_ans, open(rouge_prog, 'w'))
            json.dump(llm_ans, open(llm_prog, 'w'))
            log(f"  Progress: {i+1}/{len(test_df)}")

    json.dump(rouge_ans, open(rouge_prog, 'w'))
    json.dump(llm_ans, open(llm_prog, 'w'))

    # --- Create Gemini submissions ---
    log("\nCreating Gemini submissions...")

    # Best retrieval answer (LTR or baseline)
    def best_retrieval(i):
        q = test_qs[i].strip()
        q_sub = test_subs[i]
        D, I = global_qq_idx.search(test_emb[i:i+1], CAND_K + 5)
        candidates = []
        for j in range(CAND_K + 5):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q: continue
            candidates.append((ci, float(D[0][j])))
            if len(candidates) >= CAND_K: break
        if not candidates: return "No answer."
        if USE_LTR_PER_LANG.get(q_sub, USE_LTR) and len(candidates) >= 3:
            feats = extract_features_batch(q, test_emb[i], q_sub, candidates)
            dpred = xgb.DMatrix(np.array(feats, dtype=np.float32))
            scores = model_ltr.predict(dpred)
            return str(combined.iloc[candidates[int(np.argmax(scores))][0]]['output'])
        return str(combined.iloc[candidates[0][0]]['output'])

    # Sub 1: Retrieval ROUGE + Gemini LLM
    rows1 = []
    for i in range(len(test_df)):
        rid = str(test_df.iloc[i]['ID'])
        ret_ans = best_retrieval(i)
        rows1.append({'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': ret_ans, 'TargetRLF1': ret_ans,
            'TargetLLM': llm_ans.get(rid, ret_ans)})
    pd.DataFrame(rows1)[['ID','TargetRLF1','TargetR1F1','TargetLLM']].to_csv(
        OUTPUT_DIR / 'submission_retrieval_gemini_llm.csv', index=False)
    log("Saved: submission_retrieval_gemini_llm.csv")

    # Sub 2: Gemini ROUGE + Gemini LLM
    rows2 = []
    for i in range(len(test_df)):
        rid = str(test_df.iloc[i]['ID'])
        fallback = best_retrieval(i)
        rows2.append({'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': rouge_ans.get(rid, fallback),
            'TargetRLF1': rouge_ans.get(rid, fallback),
            'TargetLLM': llm_ans.get(rid, fallback)})
    pd.DataFrame(rows2)[['ID','TargetRLF1','TargetR1F1','TargetLLM']].to_csv(
        OUTPUT_DIR / 'submission_gemini_full.csv', index=False)
    log("Saved: submission_gemini_full.csv")

    # Sub 3: LTR ROUGE + Gemini LLM (best of both)
    rows3 = []
    for i in range(len(test_df)):
        rid = str(test_df.iloc[i]['ID'])
        ltr_ans = rows_ltr[i]['TargetR1F1']
        rows3.append({'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': ltr_ans, 'TargetRLF1': ltr_ans,
            'TargetLLM': llm_ans.get(rid, ltr_ans)})
    pd.DataFrame(rows3)[['ID','TargetRLF1','TargetR1F1','TargetLLM']].to_csv(
        OUTPUT_DIR / 'submission_ltr_gemini_llm.csv', index=False)
    log("Saved: submission_ltr_gemini_llm.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*60}")
log("DONE — ALL SUBMISSIONS READY")
log(f"{'='*60}")

log(f"\nVal scores:")
log(f"  Baseline:     R1={b_r1:.4f} RL={b_rl:.4f}")
log(f"  LTR reranked: R1={l_r1:.4f} RL={l_rl:.4f} ({l_r1-b_r1:+.4f}/{l_rl-b_rl:+.4f})")
if GEMINI_OK:
    log(f"  Gemini ext:   R1={gem_r1:.4f} RL={gem_rl:.4f}")

log(f"\nSubmissions saved to Drive:")
for f in sorted(OUTPUT_DIR.glob("submission_*.csv")):
    log(f"  → {f.name}")

log(f"\nPrevious best LB: 0.6545")
log(f"\nRecommended submit order:")
if GEMINI_OK:
    log(f"  1. submission_retrieval_gemini_llm.csv (safe: retrieval ROUGE + Gemini LLM)")
    log(f"  2. submission_ltr_gemini_llm.csv (LTR ROUGE + Gemini LLM)")
    log(f"  3. submission_gemini_full.csv (Gemini everything)")
log(f"  4. submission_ltr.csv (LTR only)")
log(f"  5. submission_perlang_best.csv (per-language retrieval)")
log(f"\nDone! Check results when you wake up.")
