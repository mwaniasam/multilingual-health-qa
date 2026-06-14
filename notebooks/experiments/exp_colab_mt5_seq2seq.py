"""
=============================================================================
COLAB PRO EXPERIMENT: mT5 Seq2Seq Generation + Retrieval Hybrid
=============================================================================
WHY THIS SHOULD WORK:
  - Retrieval is CAPPED at ~0.65 because it can only return existing answers
  - mT5 LEARNS to GENERATE answers matching the reference distribution
  - Fine-tuned on 36K Q&A pairs, it learns exact phrasing patterns
  - Should dramatically improve ROUGE-L (our biggest bottleneck at 0.56)

SCORING FORMULA (discovered!):
  Final = 0.37 * ROUGE-1 + 0.37 * ROUGE-L + 0.26 * LLM_Judge

STRATEGY:
  - Fine-tune mT5-base (580M) on question → answer
  - If time permits, try mT5-large (1.2B) with LoRA
  - Compare with retrieval baseline
  - Create HYBRID submission: best answer per column

SETUP (run in Colab):
  Cell 1:
    from google.colab import drive
    drive.mount('/content/drive')

  Cell 2:
    !pip install -q transformers datasets evaluate rouge-score accelerate peft
    !pip install -q sentencepiece protobuf
    !pip install -q sentence-transformers faiss-cpu scikit-learn

  Cell 3:
    Paste this entire script
=============================================================================
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import gc
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# CONFIG — Google Drive paths
# ============================================================
# Mount Google Drive if not already mounted
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
        log("Google Drive mounted!")
    else:
        log("Google Drive already mounted.")
except ImportError:
    log("Not running in Colab, skipping Drive mount.")

# Setup project directory in Google Drive
DRIVE_PROJECT = Path('/content/drive/MyDrive/multilingual-health-qa')
DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)

# Data directory — look for data in Drive first, then /content/data/
POSSIBLE_PATHS = [
    DRIVE_PROJECT / 'data',
    Path('/content/data/'),
    Path('/kaggle/input/multilingual-health-qa-data/'),
    Path('/kaggle/input/datasets/samuelmwania1/multilingual-health-qa-data/'),
]
DATA_DIR = None
for p in POSSIBLE_PATHS:
    if p.exists() and (p / 'Train.csv').exists():
        DATA_DIR = p
        break
if DATA_DIR is None:
    import glob
    found = glob.glob('/content/**/Train.csv', recursive=True) or glob.glob('/kaggle/**/Train.csv', recursive=True)
    if found:
        DATA_DIR = Path(found[0]).parent
    else:
        raise FileNotFoundError(
            "❌ Cannot find Train.csv!\n"
            f"Please upload Train.csv, Val.csv, Test.csv, SampleSubmission.csv to:\n"
            f"  Google Drive → My Drive → multilingual-health-qa → data\n"
            f"  (i.e. {DRIVE_PROJECT / 'data'})"
        )

# Output directory — ALWAYS save to Google Drive so results survive disconnects!
OUTPUT_DIR = DRIVE_PROJECT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log(f"Data: {DATA_DIR}")
log(f"Output: {OUTPUT_DIR} (saved to Google Drive ✅)")

# ============================================================
# LOAD & VALIDATE DATA
# ============================================================
log("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df = pd.read_csv(DATA_DIR / 'Val.csv')
test_df = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    log(f"  ✅ {name}: {len(df)} rows")

combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output']).reset_index(drop=True)
log(f"Combined corpus: {len(combined)} samples")

# GPU info
gpu_mem_gb = 0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu_name} | {gpu_mem_gb:.1f} GB")
else:
    log("⚠️ No GPU!")

# ============================================================
# PHASE 1: Fine-tune mT5-base for seq2seq generation
# ============================================================
log("\n" + "=" * 60)
log("PHASE 1: Fine-tune mT5-base for Question → Answer Generation")
log("=" * 60)

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import evaluate

# Determine model size based on GPU
if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem_gb >= 22:  # A100 (40GB) or L4 (24GB)
        MODEL_NAME = "google/mt5-large"  # 1.2B params
        if gpu_mem_gb >= 30:  # A100
            BATCH_SIZE = 4
            GRAD_ACCUM = 8  # effective batch = 32
        else:  # L4 (24GB)
            BATCH_SIZE = 2
            GRAD_ACCUM = 16  # effective batch = 32
        log(f"GPU has {gpu_mem_gb:.0f}GB — Using mT5-large (1.2B params)")
    elif gpu_mem_gb >= 14:
        MODEL_NAME = "google/mt5-base"   # V100/T4: 16GB
        BATCH_SIZE = 8
        GRAD_ACCUM = 4  # effective batch = 32
        log(f"GPU has {gpu_mem_gb:.0f}GB — Using mT5-base (580M params)")
    else:
        MODEL_NAME = "google/mt5-small"  # Low mem
        BATCH_SIZE = 16
        GRAD_ACCUM = 2
        log(f"GPU has {gpu_mem_gb:.0f}GB — Using mT5-small (300M params)")
else:
    MODEL_NAME = "google/mt5-small"
    BATCH_SIZE = 4
    GRAD_ACCUM = 4

# L4 and A100 both support bf16
USE_BF16 = torch.cuda.is_available() and gpu_mem_gb >= 22
log(f"Using bf16: {USE_BF16}")

log(f"Model: {MODEL_NAME}")
log(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    model = model.cuda()

log(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# ============================================================
# PREPARE DATASETS
# ============================================================
log("Preparing datasets...")

MAX_INPUT_LENGTH = 256   # questions are short
MAX_TARGET_LENGTH = 512  # answers can be longer

# Language prefix mapping for better multilingual performance
SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan',
    'Amh_Eth': 'Amharic',
    'Eng_Eth': 'English',
    'Eng_Gha': 'English',
    'Eng_Ken': 'English',
    'Eng_Uga': 'English',
    'Lug_Uga': 'Luganda',
    'Swa_Ken': 'Swahili',
}

def format_input(question, subset):
    """Format input with language tag for better multilingual performance."""
    lang = SUBSET_TO_LANG.get(str(subset), 'Unknown')
    return f"answer in {lang}: {question}"


def preprocess_function(examples):
    """Tokenize inputs and targets."""
    inputs = [format_input(q, s) for q, s in zip(examples['input'], examples['subset'])]
    targets = [str(a) for a in examples['output']]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['input', 'output', 'subset']].dropna())
val_dataset = Dataset.from_pandas(val_df[['input', 'output', 'subset']].dropna())

log(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# Tokenize
log("Tokenizing...")
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
log(f"  Tokenized: train={len(train_tokenized)}, val={len(val_tokenized)}")

# ============================================================
# TRAINING
# ============================================================
log("Setting up training...")

# ROUGE metric for evaluation
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in labels (padding) with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False,
    )
    return {
        "rouge1": result["rouge1"],
        "rougeL": result["rougeL"],
    }

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR / 'mt5-checkpoints'),
    num_train_epochs=5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=3e-4,  # mT5 likes higher LR
    weight_decay=0.01,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    # Evaluation — per epoch only (not every 500 steps!)
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    # Generation during eval — greedy + short max to keep eval fast
    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=1,  # greedy (beam=4 was taking 5+ hours per eval!)
    # Mixed precision
    bf16=USE_BF16,
    fp16=not USE_BF16 and torch.cuda.is_available(),
    # Logging
    logging_steps=100,
    report_to="none",
    # Memory optimization
    dataloader_num_workers=2,
    optim="adafactor",  # Recommended for T5/mT5
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

log("🚀 Starting training...")
log(f"  Total steps: ~{len(train_tokenized) * 5 // (BATCH_SIZE * GRAD_ACCUM)}")

try:
    train_result = trainer.train()
    log(f"✅ Training complete!")
    log(f"  Train loss: {train_result.training_loss:.4f}")

    # Evaluate
    eval_result = trainer.evaluate()
    log(f"  Val ROUGE-1: {eval_result.get('eval_rouge1', 'N/A')}")
    log(f"  Val ROUGE-L: {eval_result.get('eval_rougeL', 'N/A')}")

except Exception as e:
    log(f"❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    log("Attempting to save current state and generate with partial training...")

# Save the best model
try:
    trainer.save_model(str(OUTPUT_DIR / 'mt5-best'))
    tokenizer.save_pretrained(str(OUTPUT_DIR / 'mt5-best'))
    log("Model saved!")
except:
    pass


# ============================================================
# PHASE 2: Generate test predictions
# ============================================================
log("\n" + "=" * 60)
log("PHASE 2: Generate test predictions with fine-tuned mT5")
log("=" * 60)

model.eval()

def generate_answers(questions, subsets, batch_size=16, num_beams=4):
    """Generate answers for a list of questions."""
    all_answers = []
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_s = subsets[i:i+batch_size]
        inputs = [format_input(q, s) for q, s in zip(batch_q, batch_s)]

        encoded = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_length=MAX_TARGET_LENGTH,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_answers.extend(decoded)

        if (i // batch_size) % 10 == 0:
            log(f"  Generated {i+len(batch_q)}/{len(questions)}...")

    return all_answers

# Generate for validation set first (to check quality)
log("Generating val predictions...")
val_questions = val_df['input'].fillna('').tolist()
val_subsets = val_df['subset'].tolist()
val_generated = generate_answers(val_questions, val_subsets)

# Evaluate generated answers on val
from rouge_score import rouge_scorer as rs
scorer = rs.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

r1s_gen, rls_gen = [], []
for i in range(len(val_df)):
    ref = str(val_df.iloc[i]['output']).strip()
    pred = str(val_generated[i]).strip()
    if not ref or not pred:
        continue
    r = scorer.score(ref, pred)
    r1s_gen.append(r['rouge1'].fmeasure)
    rls_gen.append(r['rougeL'].fmeasure)

r1_gen = np.mean(r1s_gen)
rl_gen = np.mean(rls_gen)
log(f"[mT5 Generated] Val ROUGE-1: {r1_gen:.4f} | ROUGE-L: {rl_gen:.4f}")

# Generate for test set
log("Generating test predictions...")
test_questions = test_df['input'].fillna('').tolist()
test_subsets = test_df['subset'].tolist()
test_generated = generate_answers(test_questions, test_subsets)

# Save pure generation submission
rows = []
for i in range(len(test_df)):
    answer = str(test_generated[i]).strip() if i < len(test_generated) else ""
    if not answer:
        answer = "No answer generated."
    rows.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetRLF1': answer,
        'TargetR1F1': answer,
        'TargetLLM': answer,
    })
sub_gen = pd.DataFrame(rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_gen) == len(sample_sub)
sub_gen.to_csv(OUTPUT_DIR / 'mt5_pure_generation.csv', index=False)
log(f"✅ Saved: mt5_pure_generation.csv")


# ============================================================
# PHASE 3: Retrieval baseline for hybrid comparison
# ============================================================
log("\n" + "=" * 60)
log("PHASE 3: Retrieval baseline (for hybrid submission)")
log("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    import faiss

    # Use the pre-trained E5 (not fine-tuned, since we don't have the FT model here)
    # If you uploaded the fine-tuned model, change the path below
    FT_MODEL_PATH = str(OUTPUT_DIR / 'v4-final-model')
    if Path(FT_MODEL_PATH).exists():
        log(f"Loading fine-tuned retrieval model from {FT_MODEL_PATH}")
        retrieval_model = SentenceTransformer(FT_MODEL_PATH)
    else:
        log("No fine-tuned retrieval model found. Using base E5.")
        retrieval_model = SentenceTransformer('intfloat/multilingual-e5-base')

    questions_raw = combined['input'].fillna('').astype(str).tolist()
    answers_raw = combined['output'].fillna('').astype(str).tolist()

    # Encode corpus
    log("Encoding corpus...")
    corpus_emb = retrieval_model.encode(
        [f"query: {q}" for q in questions_raw],
        batch_size=64, show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    fidx = faiss.IndexFlatIP(corpus_emb.shape[1])
    fidx.add(corpus_emb)

    # Retrieve for val
    val_emb = retrieval_model.encode(
        [f"query: {q}" for q in val_questions],
        batch_size=64, show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    r1s_ret, rls_ret = [], []
    val_retrieved = []
    for i in range(len(val_df)):
        q = str(val_df.iloc[i]['input']).strip()
        ref = str(val_df.iloc[i]['output']).strip()
        D, I = fidx.search(val_emb[i:i+1], 10)
        pred = ''
        for j in range(10):
            cand_idx = int(I[0][j])
            if cand_idx < len(combined) and str(combined.iloc[cand_idx]['input']).strip() != q:
                pred = str(combined.iloc[cand_idx]['output'])
                break
        if not pred:
            pred = str(combined.iloc[int(I[0][0])]['output'])
        val_retrieved.append(pred)
        if ref:
            r = scorer.score(ref, pred)
            r1s_ret.append(r['rouge1'].fmeasure)
            rls_ret.append(r['rougeL'].fmeasure)

    r1_ret = np.mean(r1s_ret)
    rl_ret = np.mean(rls_ret)
    log(f"[Retrieval] Val ROUGE-1: {r1_ret:.4f} | ROUGE-L: {rl_ret:.4f}")

    # Retrieve for test
    test_emb = retrieval_model.encode(
        [f"query: {q}" for q in test_questions],
        batch_size=64, show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    test_retrieved = []
    for i in range(len(test_df)):
        q = str(test_df.iloc[i]['input']).strip()
        D, I = fidx.search(test_emb[i:i+1], 10)
        pred = ''
        for j in range(10):
            cand_idx = int(I[0][j])
            if cand_idx < len(combined) and str(combined.iloc[cand_idx]['input']).strip() != q:
                pred = str(combined.iloc[cand_idx]['output'])
                break
        if not pred:
            pred = str(combined.iloc[int(I[0][0])]['output'])
        test_retrieved.append(pred)

    HAVE_RETRIEVAL = True

except Exception as e:
    log(f"⚠️ Retrieval failed: {e}. Will use generation-only submission.")
    HAVE_RETRIEVAL = False

# ============================================================
# PHASE 4: HYBRID SUBMISSION — different answer per column!
# ============================================================
log("\n" + "=" * 60)
log("PHASE 4: Create HYBRID submissions")
log("=" * 60)

# The scoring formula is:
#   Final = 0.37 * ROUGE-1(TargetR1F1) + 0.37 * ROUGE-L(TargetRLF1) + 0.26 * LLM(TargetLLM)
#
# KEY INSIGHT: We can optimize each column independently!
# - TargetR1F1: evaluated by ROUGE-1 → best with unigram overlap → retrieval might be better
# - TargetRLF1: evaluated by ROUGE-L → best with long common subsequences → generation might help
# - TargetLLM: evaluated by LLM judge → best with comprehensive, fluent answer → generation is better

if HAVE_RETRIEVAL:
    # === HYBRID 1: Best of each on val ===
    log("\nEvaluating per-column optimization on val...")

    # For each val sample, compare generated vs retrieved on each metric
    r1_gen_wins, r1_ret_wins = 0, 0
    rl_gen_wins, rl_ret_wins = 0, 0

    for i in range(len(val_df)):
        ref = str(val_df.iloc[i]['output']).strip()
        gen = str(val_generated[i]).strip() if i < len(val_generated) else ""
        ret = val_retrieved[i] if i < len(val_retrieved) else ""

        if not ref:
            continue

        r_gen = scorer.score(ref, gen)
        r_ret = scorer.score(ref, ret)

        if r_gen['rouge1'].fmeasure > r_ret['rouge1'].fmeasure:
            r1_gen_wins += 1
        else:
            r1_ret_wins += 1

        if r_gen['rougeL'].fmeasure > r_ret['rougeL'].fmeasure:
            rl_gen_wins += 1
        else:
            rl_ret_wins += 1

    log(f"  ROUGE-1: gen wins {r1_gen_wins} vs ret wins {r1_ret_wins}")
    log(f"  ROUGE-L: gen wins {rl_gen_wins} vs ret wins {rl_ret_wins}")

    # === Create hybrid submissions ===

    # Strategy A: Generated for all (if gen is better overall)
    # Already saved as mt5_pure_generation.csv

    # Strategy B: Retrieved for ROUGE columns, Generated for LLM column
    rows_b = []
    for i in range(len(test_df)):
        ret_ans = test_retrieved[i] if i < len(test_retrieved) else ""
        gen_ans = str(test_generated[i]).strip() if i < len(test_generated) else ""
        if not ret_ans: ret_ans = gen_ans
        if not gen_ans: gen_ans = ret_ans
        rows_b.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': ret_ans,  # ROUGE-L → retrieval (exact phrases)
            'TargetR1F1': ret_ans,  # ROUGE-1 → retrieval (keyword overlap)
            'TargetLLM': gen_ans,   # LLM judge → generation (fluent, comprehensive)
        })
    sub_b = pd.DataFrame(rows_b)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_b.to_csv(OUTPUT_DIR / 'hybrid_ret_rouge_gen_llm.csv', index=False)
    log(f"✅ Saved: hybrid_ret_rouge_gen_llm.csv")

    # Strategy C: Generated for ROUGE columns, Retrieved for LLM
    rows_c = []
    for i in range(len(test_df)):
        ret_ans = test_retrieved[i] if i < len(test_retrieved) else ""
        gen_ans = str(test_generated[i]).strip() if i < len(test_generated) else ""
        if not ret_ans: ret_ans = gen_ans
        if not gen_ans: gen_ans = ret_ans
        rows_c.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': gen_ans,  # ROUGE-L → generation
            'TargetR1F1': gen_ans,  # ROUGE-1 → generation
            'TargetLLM': ret_ans,   # LLM → retrieval
        })
    sub_c = pd.DataFrame(rows_c)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_c.to_csv(OUTPUT_DIR / 'hybrid_gen_rouge_ret_llm.csv', index=False)
    log(f"✅ Saved: hybrid_gen_rouge_ret_llm.csv")

    # Strategy D: Per-sample best (oracle on val pattern)
    # Use the method that won MORE on val for each metric
    r1_use_gen = r1_gen_wins > r1_ret_wins
    rl_use_gen = rl_gen_wins > rl_ret_wins
    log(f"\n  Per-column winner: R1={'gen' if r1_use_gen else 'ret'}, RL={'gen' if rl_use_gen else 'ret'}, LLM=gen (assumed)")

    rows_d = []
    for i in range(len(test_df)):
        ret_ans = test_retrieved[i] if i < len(test_retrieved) else ""
        gen_ans = str(test_generated[i]).strip() if i < len(test_generated) else ""
        if not ret_ans: ret_ans = gen_ans
        if not gen_ans: gen_ans = ret_ans
        rows_d.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': gen_ans if rl_use_gen else ret_ans,
            'TargetR1F1': gen_ans if r1_use_gen else ret_ans,
            'TargetLLM': gen_ans,  # LLM judge always prefers generation
        })
    sub_d = pd.DataFrame(rows_d)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_d.to_csv(OUTPUT_DIR / 'hybrid_per_column_best.csv', index=False)
    log(f"✅ Saved: hybrid_per_column_best.csv")

    # Strategy E: Retrieval only (same as our Kaggle submission, for comparison)
    rows_e = []
    for i in range(len(test_df)):
        ret_ans = test_retrieved[i] if i < len(test_retrieved) else ""
        if not ret_ans: ret_ans = "No answer."
        rows_e.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': ret_ans,
            'TargetR1F1': ret_ans,
            'TargetLLM': ret_ans,
        })
    sub_e = pd.DataFrame(rows_e)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_e.to_csv(OUTPUT_DIR / 'retrieval_only_baseline.csv', index=False)
    log(f"✅ Saved: retrieval_only_baseline.csv")


# ============================================================
# PHASE 5: Simulated scoring on val to predict LB
# ============================================================
log("\n" + "=" * 60)
log("PHASE 5: Simulated LB scoring on validation set")
log("=" * 60)

def simulate_lb(r1_score, rl_score, llm_score=0.77):
    """Simulate LB score using known weights."""
    return 0.37 * r1_score + 0.37 * rl_score + 0.26 * llm_score

log(f"\nPure generation:  R1={r1_gen:.4f} RL={rl_gen:.4f} → Simulated LB: {simulate_lb(r1_gen, rl_gen):.4f}")

if HAVE_RETRIEVAL:
    log(f"Pure retrieval:   R1={r1_ret:.4f} RL={rl_ret:.4f} → Simulated LB: {simulate_lb(r1_ret, rl_ret):.4f}")

    # Simulate hybrid: best per column
    # For hybrid B: ROUGE from retrieval, LLM from generation (assume gen LLM = 0.80)
    sim_b = 0.37 * r1_ret + 0.37 * rl_ret + 0.26 * 0.80
    log(f"Hybrid B (ret+gen): R1={r1_ret:.4f} RL={rl_ret:.4f} LLM≈0.80 → Simulated LB: {sim_b:.4f}")

    # Simulate: what if generation beats retrieval on ROUGE
    best_r1 = max(r1_gen, r1_ret)
    best_rl = max(rl_gen, rl_ret)
    log(f"Best of both:     R1={best_r1:.4f} RL={best_rl:.4f} → Simulated LB: {simulate_lb(best_r1, best_rl, 0.80):.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n" + "=" * 70)
log("🏆 COLAB EXPERIMENT COMPLETE")
log("=" * 70)
log("")
log("SUBMISSIONS CREATED:")
for f in sorted(OUTPUT_DIR.glob("*.csv")):
    log(f"  → {f.name}")
log("")
log("SUBMIT PRIORITY:")
log("  1. hybrid_per_column_best.csv (optimized per metric)")
log("  2. mt5_pure_generation.csv (if gen ROUGE is high)")
log("  3. hybrid_ret_rouge_gen_llm.csv (safe hybrid)")
log("  4. retrieval_only_baseline.csv (comparison)")
log("")
log("📥 DOWNLOAD AND SUBMIT TO ZINDI!")
