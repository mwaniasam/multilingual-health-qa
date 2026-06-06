"""
=============================================================================
EXPERIMENT 12: BGE-M3 Hybrid Retrieval (Dense + Sparse + ColBERT)
=============================================================================
Run this on Kaggle with GPU T4 (16GB VRAM).

Setup:
    !pip install -q FlagEmbedding rouge-score pandas numpy tqdm

Upload to Kaggle:
    - Train.csv, Val.csv, Test.csv, SampleSubmission.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from rouge_score import rouge_scorer

# ============================================================
# CONFIG — adjust paths for Kaggle
# ============================================================
# On Kaggle: /kaggle/input/your-dataset-name/
# Locally: data/raw/
DATA_DIR = Path('/kaggle/input/multilingual-health-qa/')  # CHANGE if needed
if not DATA_DIR.exists():
    DATA_DIR = Path('data/raw/')  # fallback to local

OUTPUT_DIR = Path('/kaggle/working/')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path('submissions/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df = pd.read_csv(DATA_DIR / 'Val.csv')
test_df = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')
combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Combined: {len(combined)}")

# ============================================================
# STEP 2: Load BGE-M3 Model
# ============================================================
print("\nLoading BGE-M3 model...")
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ BGE-M3 loaded!")

# ============================================================
# STEP 3: Encode All Questions (Corpus)
# ============================================================
print("\nEncoding corpus questions...")
corpus_questions = combined['input'].fillna('').tolist()
corpus_outputs = model.encode(
    corpus_questions,
    batch_size=32,
    max_length=256,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,  # ColBERT is slow; use dense+sparse first
)
corpus_dense = corpus_outputs['dense_vecs']  # (N, 1024) numpy
corpus_sparse = corpus_outputs['lexical_weights']  # list of dicts
print(f"Corpus encoded: dense={corpus_dense.shape}")

# ============================================================
# STEP 4: Build FAISS Index for Dense Retrieval
# ============================================================
print("\nBuilding FAISS index...")
import faiss

# Normalize for cosine similarity
faiss.normalize_L2(corpus_dense)
dim = corpus_dense.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_dense)
print(f"FAISS index: {index.ntotal} vectors, dim={dim}")


# ============================================================
# STEP 5: Sparse Retrieval Function
# ============================================================
def sparse_score(q_sparse, doc_sparse):
    """Compute sparse similarity between query and document lexical weights."""
    score = 0.0
    for token, weight in q_sparse.items():
        if token in doc_sparse:
            score += weight * doc_sparse[token]
    return score


def hybrid_retrieve(query_dense, query_sparse, subset=None, top_k=10,
                    dense_weight=0.6, sparse_weight=0.4):
    """Hybrid retrieval combining dense and sparse scores."""
    # Dense retrieval: get top-50 candidates
    query_dense_norm = query_dense.copy()
    faiss.normalize_L2(query_dense_norm)
    D, I = index.search(query_dense_norm, 50)

    candidates = []
    for j in range(50):
        idx = I[0][j]
        dense_score = float(D[0][j])

        # Optional: filter by subset (language)
        if subset and combined.iloc[idx]['subset'] != subset:
            continue

        # Sparse score
        sp_score = sparse_score(query_sparse, corpus_sparse[idx])

        # Hybrid score
        hybrid = dense_weight * dense_score + sparse_weight * sp_score

        candidates.append({
            'index': idx,
            'answer': str(combined.iloc[idx]['output']),
            'question': str(combined.iloc[idx]['input']),
            'subset': combined.iloc[idx]['subset'],
            'dense_score': dense_score,
            'sparse_score': sp_score,
            'hybrid_score': hybrid,
        })

    # If subset filtering gave too few results, retry without filter
    if len(candidates) < top_k and subset:
        return hybrid_retrieve(query_dense, query_sparse, subset=None,
                               top_k=top_k, dense_weight=dense_weight,
                               sparse_weight=sparse_weight)

    # Sort by hybrid score
    candidates.sort(key=lambda x: -x['hybrid_score'])
    return candidates[:top_k]


# ============================================================
# STEP 6: Evaluate on Validation Set
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON VALIDATION SET")
print("=" * 60)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

# Encode val questions
print("Encoding val questions...")
val_questions = val_df['input'].fillna('').tolist()
val_outputs = model.encode(
    val_questions,
    batch_size=32,
    max_length=256,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
val_dense = val_outputs['dense_vecs']
val_sparse = val_outputs['lexical_weights']

# Test different weight combinations
best_weights = None
best_score = 0

for dw in [0.4, 0.5, 0.6, 0.7, 0.8]:
    sw = 1.0 - dw
    scores = []

    for idx in range(min(500, len(val_df))):  # Quick eval on 500 samples
        q = val_questions[idx]
        ref = str(val_df.iloc[idx]['output']).strip()
        subset = val_df.iloc[idx]['subset']

        q_dense = val_dense[idx:idx + 1]
        q_sparse = val_sparse[idx]

        results = hybrid_retrieve(q_dense, q_sparse, subset=subset, top_k=5,
                                  dense_weight=dw, sparse_weight=sw)

        # Skip self-match
        results = [r for r in results if r['question'] != q]
        if results:
            pred = results[0]['answer']
        else:
            pred = ''

        r = scorer.score(ref, pred)
        scores.append(r['rouge1'].fmeasure)

    avg = np.mean(scores)
    print(f"  dense={dw:.1f} sparse={sw:.1f}: ROUGE-1={avg:.4f}")

    if avg > best_score:
        best_score = avg
        best_weights = (dw, sw)

print(f"\n✅ Best weights: dense={best_weights[0]}, sparse={best_weights[1]}")
print(f"✅ Best ROUGE-1: {best_score:.4f}")

# Full val evaluation with best weights
print("\nFull val evaluation...")
all_scores = {'rouge1': [], 'rougeL': []}
for idx in tqdm(range(len(val_df)), desc="Val eval"):
    q = val_questions[idx]
    ref = str(val_df.iloc[idx]['output']).strip()
    subset = val_df.iloc[idx]['subset']

    q_dense = val_dense[idx:idx + 1]
    q_sparse = val_sparse[idx]

    results = hybrid_retrieve(q_dense, q_sparse, subset=subset, top_k=5,
                              dense_weight=best_weights[0],
                              sparse_weight=best_weights[1])
    results = [r for r in results if r['question'] != q]

    pred = results[0]['answer'] if results else ''

    r = scorer.score(ref, pred)
    all_scores['rouge1'].append(r['rouge1'].fmeasure)
    all_scores['rougeL'].append(r['rougeL'].fmeasure)

print(f"\nFull val ROUGE-1: {np.mean(all_scores['rouge1']):.4f}")
print(f"Full val ROUGE-L: {np.mean(all_scores['rougeL']):.4f}")
print(f"Compare to E5-base: 0.5219")

# ============================================================
# STEP 7: Generate Test Submission
# ============================================================
print("\n" + "=" * 60)
print("GENERATING TEST SUBMISSION")
print("=" * 60)

print("Encoding test questions...")
test_questions = test_df['input'].fillna('').tolist()
test_outputs_enc = model.encode(
    test_questions,
    batch_size=32,
    max_length=256,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
test_dense = test_outputs_enc['dense_vecs']
test_sparse = test_outputs_enc['lexical_weights']

rows = []
for idx in tqdm(range(len(test_df)), desc="Test submission"):
    q = test_questions[idx]
    subset = test_df.iloc[idx]['subset']

    q_dense = test_dense[idx:idx + 1]
    q_sparse = test_sparse[idx]

    results = hybrid_retrieve(q_dense, q_sparse, subset=subset, top_k=3,
                              dense_weight=best_weights[0],
                              sparse_weight=best_weights[1])

    answer = results[0]['answer'] if results else "Health information not available."

    rows.append({
        'ID': test_df.iloc[idx]['ID'],
        'TargetRLF1': answer,
        'TargetR1F1': answer,
        'TargetLLM': answer,
    })

sub = pd.DataFrame(rows)
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)

path = OUTPUT_DIR / 'exp12_bge_m3_hybrid.csv'
sub.to_csv(path, index=False)
print(f"\n✅ Saved: {path}")
print(f"Shape: {sub.shape}")
print(f"\nDOWNLOAD THIS FILE AND SUBMIT TO ZINDI!")
print(f"Comment: Experiment 12: BGE-M3 hybrid retrieval (dense={best_weights[0]:.1f} + sparse={best_weights[1]:.1f}). Auto-tuned weight ratio on val set.")
