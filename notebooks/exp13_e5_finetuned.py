"""
EXPERIMENT 13: Fine-Tuned E5-base — FIXED FOR T4 MEMORY
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use SINGLE GPU, avoid DataParallel OOM

import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

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
print(f"Combined: {len(combined)} samples")

# ============================================================
# STEP 2: Prepare Training Data
# ============================================================
print("\nPreparing contrastive training data...")
train_examples = []
for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Building pairs"):
    q = str(row['input']).strip()
    a = str(row['output']).strip()
    if q and a:
        train_examples.append(InputExample(texts=[f"query: {q}", f"passage: {a}"]))
print(f"Training examples: {len(train_examples)}")

# ============================================================
# STEP 3: Fine-Tune E5-base
# ============================================================
print("\nLoading E5-base model...")
model = SentenceTransformer('intfloat/multilingual-e5-base', device='cuda:0')

train_loss = losses.MultipleNegativesRankingLoss(model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)  # Small batch for T4

print(f"Training for 3 epochs, batch_size=8, steps={len(train_dataloader) * 3}")
print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5-base-finetuned'),
    use_amp=True,  # Mixed precision to save memory
)
print("✅ Fine-tuning complete!")

# ============================================================
# STEP 4: Build FAISS Index
# ============================================================
print("\nBuilding FAISS index with fine-tuned model...")
corpus_questions = [f"query: {q}" for q in combined['input'].fillna('').tolist()]
corpus_embeddings = model.encode(
    corpus_questions, batch_size=64,
    show_progress_bar=True, normalize_embeddings=True,
)
corpus_embeddings = corpus_embeddings.astype(np.float32)
print(f"Corpus: {corpus_embeddings.shape}, dtype={corpus_embeddings.dtype}")

index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)
print(f"FAISS index: {index.ntotal} vectors")

# ============================================================
# STEP 5: Evaluate on Val
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON VALIDATION SET")
print("=" * 60)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

val_questions_raw = val_df['input'].fillna('').tolist()
val_questions = [f"query: {q}" for q in val_questions_raw]
val_embeddings = model.encode(val_questions, batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True)
val_embeddings = val_embeddings.astype(np.float32)

rouge1_scores, rougeL_scores = [], []
for idx in tqdm(range(len(val_df)), desc="Val eval"):
    q = str(val_df.iloc[idx]['input']).strip()
    ref = str(val_df.iloc[idx]['output']).strip()
    q_emb = val_embeddings[idx:idx + 1]
    D, I = index.search(q_emb, 10)
    pred = ''
    for j in range(10):
        if str(combined.iloc[I[0][j]]['input']).strip() != q:
            pred = str(combined.iloc[I[0][j]]['output'])
            break
    if not pred:
        pred = str(combined.iloc[I[0][0]]['output'])
    r = scorer.score(ref, pred)
    rouge1_scores.append(r['rouge1'].fmeasure)
    rougeL_scores.append(r['rougeL'].fmeasure)

print(f"\n{'='*60}")
print(f"Fine-tuned E5 ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"Fine-tuned E5 ROUGE-L: {np.mean(rougeL_scores):.4f}")
print(f"Baseline E5 (no FT):   0.5219")
print(f"Improvement:           {np.mean(rouge1_scores) - 0.5219:+.4f}")
print(f"{'='*60}")

# ============================================================
# STEP 6: Generate Test Submission
# ============================================================
print("\nGenerating test submission...")
test_questions = [f"query: {q}" for q in test_df['input'].fillna('').tolist()]
test_embeddings = model.encode(test_questions, batch_size=64,
                               normalize_embeddings=True, show_progress_bar=True)
test_embeddings = test_embeddings.astype(np.float32)

rows = []
for idx in tqdm(range(len(test_df)), desc="Test submission"):
    q_emb = test_embeddings[idx:idx + 1]
    D, I = index.search(q_emb, 3)
    answer = str(combined.iloc[I[0][0]]['output'])
    rows.append({
        'ID': test_df.iloc[idx]['ID'],
        'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
    })

sub = pd.DataFrame(rows)
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)

path = OUTPUT_DIR / 'exp13_e5_finetuned.csv'
sub.to_csv(path, index=False)
print(f"\n✅ Saved: {path}")
print(f"Shape: {sub.shape}")
print(f"\n📥 DOWNLOAD AND SUBMIT TO ZINDI!")
print(f"Comment: Experiment 13: E5-base fine-tuned with contrastive learning (MNRL) on {len(combined)} health QA pairs, 3 epochs, AMP.")

model.save(str(OUTPUT_DIR / 'e5-base-finetuned-final'))
print("✅ Model saved")
