"""
EXPERIMENT 12: BGE-M3 Hybrid Retrieval — FULLY FIXED
"""
import os
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer

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
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Combined: {len(combined)}")

# ============================================================
# STEP 2: Load BGE-M3
# ============================================================
print("\nLoading BGE-M3 model...")
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("✅ BGE-M3 loaded!")

# ============================================================
# STEP 3: Encode Corpus
# ============================================================
print("\nEncoding corpus questions...")
corpus_questions = combined['input'].fillna('').tolist()
corpus_out = model.encode(corpus_questions, batch_size=32, max_length=256,
                          return_dense=True, return_sparse=True, return_colbert_vecs=False)
corpus_dense = corpus_out['dense_vecs'].astype(np.float32)  # FIX: force float32
corpus_sparse = corpus_out['lexical_weights']
print(f"Corpus: {corpus_dense.shape}, dtype={corpus_dense.dtype}")

# ============================================================
# STEP 4: Build FAISS Index
# ============================================================
print("\nBuilding FAISS index...")
faiss.normalize_L2(corpus_dense)
index = faiss.IndexFlatIP(corpus_dense.shape[1])
index.add(corpus_dense)
print(f"FAISS index: {index.ntotal} vectors")

# ============================================================
# STEP 5: Retrieval Functions
# ============================================================
def sparse_score(q_sp, doc_sp):
    return sum(w * doc_sp.get(t, 0.0) for t, w in q_sp.items())

def hybrid_retrieve(q_dense, q_sparse, q_text, top_k=5, dense_w=0.6, sparse_w=0.4):
    q = q_dense.astype(np.float32).copy()  # FIX: always float32
    faiss.normalize_L2(q)
    D, I = index.search(q, 50)
    candidates = []
    for j in range(50):
        idx = int(I[0][j])
        cq = str(combined.iloc[idx]['input']).strip()
        if cq == q_text:
            continue  # skip self-match
        ds = float(D[0][j])
        ss = sparse_score(q_sparse, corpus_sparse[idx])
        candidates.append({
            'answer': str(combined.iloc[idx]['output']),
            'score': dense_w * ds + sparse_w * ss,
        })
        if len(candidates) >= top_k:
            break
    candidates.sort(key=lambda x: -x['score'])
    return candidates

# ============================================================
# STEP 6: Evaluate on Val (tune weights)
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON VALIDATION SET")
print("=" * 60)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

print("Encoding val questions...")
val_questions = val_df['input'].fillna('').tolist()
val_out = model.encode(val_questions, batch_size=32, max_length=256,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False)
val_dense = val_out['dense_vecs'].astype(np.float32)  # FIX
val_sparse = val_out['lexical_weights']

best_weights = (0.6, 0.4)
best_score = 0

for dw in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    sw = 1.0 - dw
    scores = []
    for idx in range(min(500, len(val_df))):
        q = val_questions[idx]
        ref = str(val_df.iloc[idx]['output']).strip()
        results = hybrid_retrieve(val_dense[idx:idx+1], val_sparse[idx], q,
                                  top_k=3, dense_w=dw, sparse_w=sw)
        pred = results[0]['answer'] if results else ''
        r = scorer.score(ref, pred)
        scores.append(r['rouge1'].fmeasure)
    avg = np.mean(scores)
    print(f"  dense={dw:.1f} sparse={sw:.1f}: ROUGE-1={avg:.4f}")
    if avg > best_score:
        best_score = avg
        best_weights = (dw, sw)

print(f"\n✅ Best: dense={best_weights[0]}, sparse={best_weights[1]}, ROUGE-1={best_score:.4f}")

# Full val eval
print("\nFull val evaluation...")
r1_all, rl_all = [], []
for idx in tqdm(range(len(val_df)), desc="Val eval"):
    q = val_questions[idx]
    ref = str(val_df.iloc[idx]['output']).strip()
    results = hybrid_retrieve(val_dense[idx:idx+1], val_sparse[idx], q,
                              top_k=3, dense_w=best_weights[0], sparse_w=best_weights[1])
    pred = results[0]['answer'] if results else ''
    r = scorer.score(ref, pred)
    r1_all.append(r['rouge1'].fmeasure)
    rl_all.append(r['rougeL'].fmeasure)

print(f"\n{'='*60}")
print(f"FULL VAL ROUGE-1: {np.mean(r1_all):.4f}")
print(f"FULL VAL ROUGE-L: {np.mean(rl_all):.4f}")
print(f"E5-base baseline: 0.5219")
print(f"{'='*60}")

# ============================================================
# STEP 7: Generate Test Submission
# ============================================================
print("\nEncoding test questions...")
test_questions = test_df['input'].fillna('').tolist()
test_out = model.encode(test_questions, batch_size=32, max_length=256,
                        return_dense=True, return_sparse=True, return_colbert_vecs=False)
test_dense = test_out['dense_vecs'].astype(np.float32)  # FIX
test_sparse = test_out['lexical_weights']

rows = []
for idx in tqdm(range(len(test_df)), desc="Test submission"):
    q = test_questions[idx]
    results = hybrid_retrieve(test_dense[idx:idx+1], test_sparse[idx], q,
                              top_k=3, dense_w=best_weights[0], sparse_w=best_weights[1])
    answer = results[0]['answer'] if results else "No answer available."
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
print(f"\n✅ DONE! Saved: {path}")
print(f"Shape: {sub.shape}")
print(f"\n📥 DOWNLOAD THIS FILE AND SUBMIT TO ZINDI!")
print(f"Comment: Experiment 12: BGE-M3 hybrid retrieval (dense={best_weights[0]:.1f} + sparse={best_weights[1]:.1f}). SOTA multilingual retrieval model with auto-tuned dense/sparse weights on val.")
