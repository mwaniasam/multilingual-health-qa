"""
=============================================================================
EXPERIMENT 13: Fine-Tuned E5-base with Contrastive Learning
=============================================================================
Run this on Kaggle with GPU T4 (16GB VRAM).

Setup:
    !pip install -q sentence-transformers faiss-gpu rouge-score pandas numpy tqdm

Upload to Kaggle:
    - Train.csv, Val.csv, Test.csv, SampleSubmission.csv
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path('/kaggle/input/multilingual-health-qa/')
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
# STEP 2: Prepare Training Data with Hard Negatives
# ============================================================
print("\nPreparing contrastive training data...")
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader

# Create (question, answer) positive pairs
# For MultipleNegativesRankingLoss, in-batch negatives are automatic
train_examples = []
for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Building pairs"):
    q = str(row['input']).strip()
    a = str(row['output']).strip()
    if q and a:
        # E5 format: prefix with query/passage
        train_examples.append(InputExample(
            texts=[f"query: {q}", f"passage: {a}"]
        ))

print(f"Training examples: {len(train_examples)}")

# ============================================================
# STEP 3: Fine-Tune E5-base
# ============================================================
print("\nLoading E5-base model...")
model = SentenceTransformer('intfloat/multilingual-e5-base')

# MultipleNegativesRankingLoss: for each (q, a+) pair,
# all other a's in the batch are treated as negatives
train_loss = losses.MultipleNegativesRankingLoss(model)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

print(f"Training for 3 epochs with batch_size=32...")
print(f"Total steps: {len(train_dataloader) * 3}")

# Use the old-style fit() for simplicity
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=str(OUTPUT_DIR / 'e5-base-finetuned'),
)

print("✅ Fine-tuning complete!")

# ============================================================
# STEP 4: Build FAISS Index with Fine-Tuned Model
# ============================================================
print("\nBuilding FAISS index with fine-tuned model...")
import faiss

# Encode corpus
corpus_questions = [f"query: {q}" for q in combined['input'].fillna('').tolist()]
corpus_embeddings = model.encode(
    corpus_questions,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"Corpus embeddings: {corpus_embeddings.shape}")

# Build index
dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_embeddings.astype(np.float32))
print(f"FAISS index: {index.ntotal} vectors")

# ============================================================
# STEP 5: Evaluate on Validation Set
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON VALIDATION SET")
print("=" * 60)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

val_questions = [f"query: {q}" for q in val_df['input'].fillna('').tolist()]
val_embeddings = model.encode(val_questions, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True)

rouge1_scores = []
for idx in tqdm(range(len(val_df)), desc="Val eval"):
    q = str(val_df.iloc[idx]['input']).strip()
    ref = str(val_df.iloc[idx]['output']).strip()

    q_emb = val_embeddings[idx:idx + 1]
    D, I = index.search(q_emb, 10)

    # Skip self-match
    for j in range(10):
        cand_q = str(combined.iloc[I[0][j]]['input']).strip()
        if cand_q != q:
            pred = str(combined.iloc[I[0][j]]['output'])
            break
    else:
        pred = str(combined.iloc[I[0][0]]['output'])

    r = scorer.score(ref, pred)
    rouge1_scores.append(r['rouge1'].fmeasure)

print(f"\nFine-tuned E5 ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"Baseline E5 (no FT):    0.5219")
print(f"Improvement:            {np.mean(rouge1_scores) - 0.5219:+.4f}")

# ============================================================
# STEP 6: Generate Test Submission
# ============================================================
print("\n" + "=" * 60)
print("GENERATING TEST SUBMISSION")
print("=" * 60)

test_questions = [f"query: {q}" for q in test_df['input'].fillna('').tolist()]
test_embeddings = model.encode(test_questions, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=True)

rows = []
for idx in tqdm(range(len(test_df)), desc="Test submission"):
    q_emb = test_embeddings[idx:idx + 1]
    D, I = index.search(q_emb, 3)
    answer = str(combined.iloc[I[0][0]]['output'])

    rows.append({
        'ID': test_df.iloc[idx]['ID'],
        'TargetRLF1': answer,
        'TargetR1F1': answer,
        'TargetLLM': answer,
    })

sub = pd.DataFrame(rows)
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)

path = OUTPUT_DIR / 'exp13_e5_finetuned.csv'
sub.to_csv(path, index=False)
print(f"\n✅ Saved: {path}")
print(f"Shape: {sub.shape}")
print(f"\nDOWNLOAD THIS FILE AND SUBMIT TO ZINDI!")
print(f"Comment: Experiment 13: E5-base fine-tuned with contrastive learning (MultipleNegativesRankingLoss) on 36K health QA pairs, 3 epochs.")

# Also save the model for later use
model.save(str(OUTPUT_DIR / 'e5-base-finetuned-final'))
print("✅ Model saved for reuse")
