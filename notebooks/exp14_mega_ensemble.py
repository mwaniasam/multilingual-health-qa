"""
=============================================================================
EXPERIMENT 14: BGE-M3 Fine-Tuned Hybrid + Mega Ensemble
=============================================================================
Run this on Kaggle AFTER experiments 12 and 13.
Upload the saved e5-finetuned model from exp13 as a dataset.

Setup:
    !pip install -q FlagEmbedding sentence-transformers faiss-gpu rouge-score pandas numpy tqdm scikit-learn

Upload to Kaggle:
    - Train.csv, Val.csv, Test.csv, SampleSubmission.csv
    - e5-base-finetuned-final/ (from exp13 output)
"""

import os
import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# ============================================================
# STEP 1: Load Data
# ============================================================
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df = pd.read_csv(DATA_DIR / 'Val.csv')
test_df = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')
combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])

# ============================================================
# STEP 2: Load ALL Retrieval Models
# ============================================================

# --- Model 1: BGE-M3 ---
print("Loading BGE-M3...")
from FlagEmbedding import BGEM3FlagModel
bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# --- Model 2: Fine-tuned E5 ---
print("Loading fine-tuned E5...")
from sentence_transformers import SentenceTransformer

# Try loading from previous experiment output
e5_ft_path = '/kaggle/input/e5-finetuned/e5-base-finetuned-final'
if not Path(e5_ft_path).exists():
    e5_ft_path = str(OUTPUT_DIR / 'e5-base-finetuned-final')
if not Path(e5_ft_path).exists():
    print("⚠️ Fine-tuned E5 not found, using base E5")
    e5_ft_path = 'intfloat/multilingual-e5-base'
e5_model = SentenceTransformer(e5_ft_path)

# --- BM25 (TF-IDF) ---
print("Building BM25 indexes...")
bm25_indexes = {}
for subset in combined['subset'].unique():
    sub_data = combined[combined['subset'] == subset].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000,
                          sublinear_tf=True, min_df=1)
    matrix = vec.fit_transform(sub_data['input'].fillna(''))
    bm25_indexes[subset] = {'vectorizer': vec, 'matrix': matrix, 'data': sub_data}

# ============================================================
# STEP 3: Encode Corpus with All Models
# ============================================================
questions = combined['input'].fillna('').tolist()

print("Encoding with BGE-M3...")
bge_out = bge_model.encode(questions, batch_size=32, max_length=256,
                            return_dense=True, return_sparse=True,
                            return_colbert_vecs=False)
bge_dense = bge_out['dense_vecs']
bge_sparse = bge_out['lexical_weights']
faiss.normalize_L2(bge_dense)
bge_index = faiss.IndexFlatIP(bge_dense.shape[1])
bge_index.add(bge_dense)

print("Encoding with fine-tuned E5...")
e5_questions = [f"query: {q}" for q in questions]
e5_embeddings = e5_model.encode(e5_questions, batch_size=64,
                                 normalize_embeddings=True,
                                 show_progress_bar=True)
e5_index = faiss.IndexFlatIP(e5_embeddings.shape[1])
e5_index.add(e5_embeddings.astype(np.float32))

print("All models ready!")


# ============================================================
# STEP 4: Mega RRF Retrieval
# ============================================================
def sparse_score(q_sparse, doc_sparse):
    score = 0.0
    for token, weight in q_sparse.items():
        if token in doc_sparse:
            score += weight * doc_sparse[token]
    return score


def mega_rrf_retrieve(q_text, q_bge_dense, q_bge_sparse, q_e5_emb, subset, k=60):
    """Reciprocal Rank Fusion across BGE-M3-dense, BGE-M3-sparse, E5-ft, BM25."""
    answer_scores = {}  # answer_text -> RRF score

    # BGE-M3 dense top-20
    q_dense = q_bge_dense.copy()
    faiss.normalize_L2(q_dense)
    D_b, I_b = bge_index.search(q_dense, 20)
    for rank in range(20):
        idx = I_b[0][rank]
        ans = str(combined.iloc[idx]['output'])
        cq = str(combined.iloc[idx]['input'])
        if cq != q_text:
            answer_scores[ans] = answer_scores.get(ans, 0) + 1.0 / (k + rank + 1)

    # BGE-M3 sparse reranking of top dense candidates
    sparse_candidates = []
    for rank in range(min(20, len(I_b[0]))):
        idx = I_b[0][rank]
        sp = sparse_score(q_bge_sparse, bge_sparse[idx])
        sparse_candidates.append((idx, sp))
    sparse_candidates.sort(key=lambda x: -x[1])
    for rank, (idx, _) in enumerate(sparse_candidates[:20]):
        ans = str(combined.iloc[idx]['output'])
        cq = str(combined.iloc[idx]['input'])
        if cq != q_text:
            answer_scores[ans] = answer_scores.get(ans, 0) + 1.0 / (k + rank + 1)

    # E5 fine-tuned top-20
    D_e, I_e = e5_index.search(q_e5_emb, 20)
    for rank in range(20):
        idx = I_e[0][rank]
        ans = str(combined.iloc[idx]['output'])
        cq = str(combined.iloc[idx]['input'])
        if cq != q_text:
            answer_scores[ans] = answer_scores.get(ans, 0) + 1.0 / (k + rank + 1)

    # BM25 top-20
    if subset in bm25_indexes:
        idx_b = bm25_indexes[subset]
        sims = cosine_similarity(idx_b['vectorizer'].transform([q_text]),
                                  idx_b['matrix']).flatten()
        for rank, j in enumerate(np.argsort(sims)[::-1][:20]):
            cq = idx_b['data'].iloc[j]['input']
            if cq != q_text:
                ans = str(idx_b['data'].iloc[j]['output'])
                answer_scores[ans] = answer_scores.get(ans, 0) + 1.0 / (k + rank + 1)

    # Return top answer by RRF score
    if answer_scores:
        sorted_answers = sorted(answer_scores.items(), key=lambda x: -x[1])
        return sorted_answers[0][0], sorted_answers
    else:
        return "", []


# ============================================================
# STEP 5: Evaluate on Val
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING MEGA ENSEMBLE ON VALIDATION SET")
print("=" * 60)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

# Encode val
val_qs = val_df['input'].fillna('').tolist()
print("Encoding val with BGE-M3...")
val_bge = bge_model.encode(val_qs, batch_size=32, max_length=256,
                            return_dense=True, return_sparse=True,
                            return_colbert_vecs=False)
val_bge_dense = val_bge['dense_vecs']
val_bge_sparse = val_bge['lexical_weights']

print("Encoding val with E5-ft...")
val_e5_qs = [f"query: {q}" for q in val_qs]
val_e5_emb = e5_model.encode(val_e5_qs, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True)

rouge1_scores = []
for idx in tqdm(range(len(val_df)), desc="Val eval"):
    q = val_qs[idx]
    ref = str(val_df.iloc[idx]['output']).strip()
    subset = val_df.iloc[idx]['subset']

    answer, _ = mega_rrf_retrieve(
        q, val_bge_dense[idx:idx + 1], val_bge_sparse[idx],
        val_e5_emb[idx:idx + 1], subset
    )

    r = scorer.score(ref, answer)
    rouge1_scores.append(r['rouge1'].fmeasure)

print(f"\nMega Ensemble ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"E5-base baseline:     0.5219")
print(f"Smart Selector:       0.5727")

# ============================================================
# STEP 6: Generate Test Submission
# ============================================================
print("\n" + "=" * 60)
print("GENERATING TEST SUBMISSION")
print("=" * 60)

test_qs = test_df['input'].fillna('').tolist()
print("Encoding test with BGE-M3...")
test_bge = bge_model.encode(test_qs, batch_size=32, max_length=256,
                             return_dense=True, return_sparse=True,
                             return_colbert_vecs=False)
test_bge_dense = test_bge['dense_vecs']
test_bge_sparse = test_bge['lexical_weights']

print("Encoding test with E5-ft...")
test_e5_qs = [f"query: {q}" for q in test_qs]
test_e5_emb = e5_model.encode(test_e5_qs, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True)

rows = []
for idx in tqdm(range(len(test_df)), desc="Test submission"):
    q = test_qs[idx]
    subset = test_df.iloc[idx]['subset']

    answer, _ = mega_rrf_retrieve(
        q, test_bge_dense[idx:idx + 1], test_bge_sparse[idx],
        test_e5_emb[idx:idx + 1], subset
    )
    if not answer:
        answer = "Health information not available."

    rows.append({
        'ID': test_df.iloc[idx]['ID'],
        'TargetRLF1': answer,
        'TargetR1F1': answer,
        'TargetLLM': answer,
    })

sub = pd.DataFrame(rows)
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)

path = OUTPUT_DIR / 'exp14_mega_ensemble.csv'
sub.to_csv(path, index=False)
print(f"\n✅ Saved: {path}")
print(f"Shape: {sub.shape}")
print(f"\nDOWNLOAD THIS FILE AND SUBMIT TO ZINDI!")
print(f"Comment: Experiment 14: Mega RRF ensemble — BGE-M3 (dense+sparse) + fine-tuned E5 + BM25. Four retrieval signals fused with Reciprocal Rank Fusion (k=60).")
