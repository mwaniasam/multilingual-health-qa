"""
=============================================================================
Qwen2.5-7B RAG Reader — Fine-tuned on YOUR data with AfriE5 retrieval
=============================================================================
NEW SESSION — everything from scratch.

What this does:
1. Loads AfriE5 from Drive (your trained retriever)
2. Builds retrieval index
3. Fine-tunes Qwen2.5-7B-Instruct to COPY retrieved answers in the right style
4. Generates test submission

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm scikit-learn peft trl accelerate datasets
Cell 3: Paste this script
=============================================================================
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import inspect
import numpy as np
import pandas as pd
import torch
import faiss
import gc
import traceback
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
log(f"Data: {DATA_DIR}")
log(f"Output: {OUTPUT_DIR}")

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
# STEP 1: LOAD AfriE5 + BUILD INDEX
# ============================================================
log(f"\n{'='*60}")
log("STEP 1: Load AfriE5 + Build retrieval index")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer

PREFIX = "query: "

if AFRIE5_DIR and AFRIE5_DIR.exists():
    bienc = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0')
    log(f"AfriE5 loaded from Drive: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")
else:
    bienc = SentenceTransformer('McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
    log(f"Loaded from HuggingFace: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

log("Encoding corpus...")
corpus_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
fidx = faiss.IndexFlatIP(corpus_emb.shape[1])
fidx.add(corpus_emb)
log(f"Index: {corpus_emb.shape}")

log("Encoding val...")
val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

log("Encoding train for RAG data...")
train_qs_text = train_df['input'].fillna('').astype(str).tolist()
train_as_text = train_df['output'].fillna('').astype(str).tolist()
train_subsets = train_df['subset'].fillna('').astype(str).tolist()
train_q_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in train_qs_text],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

log("Encoding test...")
test_inputs_list = test_df['input'].fillna('').astype(str).tolist()
test_subsets_list = test_df['subset'].fillna('').astype(str).tolist()
test_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in test_inputs_list],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

# Free bi-encoder from GPU — we have all embeddings
bienc.cpu()
gc.collect(); torch.cuda.empty_cache()
log("All embeddings computed. Bi-encoder moved to CPU.")

# ============================================================
# STEP 2: BUILD RAG TRAINING DATA
# ============================================================
log(f"\n{'='*60}")
log("STEP 2: Build RAG training data")
log(f"{'='*60}")

SYSTEM_PROMPT = (
    "You are a multilingual health QA assistant. "
    "Answer the question using the provided reference answers. "
    "Keep the exact same wording, phrasing, and medical terminology as the references. "
    "Do NOT add information not in the references. "
    "Answer in the same language as the question."
)

train_texts = []
for i in tqdm(range(len(train_df)), desc="Building train"):
    q = train_qs_text[i]
    ref_answer = train_as_text[i]
    lang = SUBSET_TO_LANG.get(train_subsets[i], train_subsets[i])
    if not q.strip() or not ref_answer.strip():
        continue

    D, I = fidx.search(train_q_emb[i:i+1], 10)
    contexts = []
    for j in range(10):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        cq = str(combined.iloc[ci]['input']).strip()
        ca = str(combined.iloc[ci]['output']).strip()
        if cq == q.strip(): continue
        if ca == ref_answer.strip(): continue
        contexts.append(ca)
        if len(contexts) >= 3: break

    if not contexts:
        context_str = "No reference answers available."
    else:
        context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)])

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question ({lang}): {q}\n\n"
        f"Reference answers:\n{context_str}<|im_end|>\n"
        f"<|im_start|>assistant\n{ref_answer}<|im_end|>"
    )
    train_texts.append(text)

log(f"Training examples: {len(train_texts)}")

val_texts = []
for i in range(min(1000, len(val_df))):
    q = val_qs[i]
    ref_answer = str(val_df.iloc[i]['output']).strip()
    lang = SUBSET_TO_LANG.get(str(val_df.iloc[i]['subset']), str(val_df.iloc[i]['subset']))
    if not q.strip() or not ref_answer.strip(): continue

    D, I = fidx.search(val_emb[i:i+1], 10)
    contexts = []
    for j in range(10):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q.strip(): continue
        ca = str(combined.iloc[ci]['output']).strip()
        if ca == ref_answer.strip(): continue
        contexts.append(ca)
        if len(contexts) >= 3: break

    context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)]) if contexts else "No reference answers available."
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question ({lang}): {q}\n\n"
        f"Reference answers:\n{context_str}<|im_end|>\n"
        f"<|im_start|>assistant\n{ref_answer}<|im_end|>"
    )
    val_texts.append(text)

log(f"Val examples: {len(val_texts)}")

# Free training embeddings
del train_q_emb
gc.collect()

# ============================================================
# STEP 3: FINE-TUNE Qwen2.5-7B with LoRA
# ============================================================
log(f"\n{'='*60}")
log("STEP 3: Fine-tune Qwen2.5-7B-Instruct")
log(f"{'='*60}")

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset

import trl
log(f"trl version: {trl.__version__}")

# Print what SFTTrainer accepts so we know
sft_trainer_params = list(inspect.signature(SFTTrainer.__init__).parameters.keys())
log(f"SFTTrainer accepts: {sft_trainer_params}")

READER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
log(f"\nLoading {READER_MODEL}...")

tokenizer = AutoTokenizer.from_pretrained(READER_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    reader_model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    log("Using Flash Attention 2")
except Exception as e:
    log(f"Flash Attention unavailable ({type(e).__name__}), using default")
    reader_model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

log(f"Model: {sum(p.numel() for p in reader_model.parameters())/1e9:.1f}B params")

lora_config = LoraConfig(
    r=64, lora_alpha=128, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

# Use TrainingArguments (universal, works with ALL trl versions)
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / 'qwen-rag-reader'),
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    save_total_limit=1,
    dataloader_num_workers=2,
)

# Build SFTTrainer — only pass params it actually accepts
sft_kwargs = {
    "model": reader_model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    "peft_config": lora_config,
}

optional = {
    "processing_class": tokenizer,
    "tokenizer": tokenizer,
    "max_seq_length": 768,
    "dataset_text_field": "text",
    "packing": False,
}

for param, value in optional.items():
    if param in sft_trainer_params:
        sft_kwargs[param] = value
        log(f"  + {param}")

# Avoid passing both tokenizer and processing_class
if "processing_class" in sft_kwargs and "tokenizer" in sft_kwargs:
    del sft_kwargs["tokenizer"]

log(f"\nFinal SFTTrainer kwargs: {list(sft_kwargs.keys())}")
trainer = SFTTrainer(**sft_kwargs)
log("Starting training...")
trainer.train()
log("Training complete!")

trainer.save_model(str(OUTPUT_DIR / 'qwen-rag-lora'))
tokenizer.save_pretrained(str(OUTPUT_DIR / 'qwen-rag-lora'))
log("LoRA adapter saved to Drive!")

reader_model = trainer.model
reader_model.eval()

# ============================================================
# STEP 4: GENERATE TEST SUBMISSION
# ============================================================
log(f"\n{'='*60}")
log("STEP 4: Generate test submission")
log(f"{'='*60}")

gen_rows = []
for i in tqdm(range(len(test_df)), desc="Generating"):
    q = test_inputs_list[i]
    lang = SUBSET_TO_LANG.get(test_subsets_list[i], test_subsets_list[i])

    D, I = fidx.search(test_emb[i:i+1], 10)
    contexts = []
    for j in range(10):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q.strip(): continue
        contexts.append(str(combined.iloc[ci]['output']))
        if len(contexts) >= 3: break

    context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)]) if contexts else "No reference answers available."

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question ({lang}): {q}\n\n"
        f"Reference answers:\n{context_str}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700)
    inputs = {k: v.to(reader_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = reader_model.generate(
            **inputs, max_new_tokens=512, temperature=0.3,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    for marker in ['<|im_end|>', '<|endoftext|>', '<|im_start|>']:
        if marker in gen_text:
            gen_text = gen_text.split(marker)[0].strip()
    if not gen_text:
        gen_text = "No answer available."

    gen_rows.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetRLF1': gen_text, 'TargetR1F1': gen_text, 'TargetLLM': gen_text,
    })

    if (i + 1) % 500 == 0:
        log(f"  Progress: {i+1}/{len(test_df)}")
        pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']].to_csv(
            OUTPUT_DIR / 'submission_rag_reader_PARTIAL.csv', index=False)

sub_gen = pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_gen) == len(sample_sub), f"Length mismatch: {len(sub_gen)} vs {len(sample_sub)}"
for col in ['TargetRLF1', 'TargetR1F1', 'TargetLLM']:
    sub_gen[col] = sub_gen[col].fillna("No answer available.")
    sub_gen[col] = sub_gen[col].replace('', "No answer available.")
sub_gen.to_csv(OUTPUT_DIR / 'submission_rag_reader.csv', index=False)
log("Saved: submission_rag_reader.csv")

# ============================================================
# STEP 5: EVALUATE ON VAL
# ============================================================
log(f"\n{'='*60}")
log("STEP 5: Evaluate on val")
log(f"{'='*60}")

gen_r1s, gen_rls = [], []
baseline_r1s, baseline_rls = [], []
sample_n = min(300, len(val_df))

for i in tqdm(range(sample_n), desc="Val eval"):
    q = val_qs[i]
    ref = str(val_df.iloc[i]['output']).strip()
    lang = SUBSET_TO_LANG.get(str(val_df.iloc[i]['subset']), str(val_df.iloc[i]['subset']))
    if not ref: continue

    D, I = fidx.search(val_emb[i:i+1], 10)
    contexts = []
    baseline_answer = ''
    for j in range(10):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q.strip(): continue
        ca = str(combined.iloc[ci]['output'])
        if not baseline_answer:
            baseline_answer = ca
        contexts.append(ca)
        if len(contexts) >= 3: break

    # Baseline: top-1 retrieval
    if baseline_answer:
        r_base = scorer.score(ref, baseline_answer)
        baseline_r1s.append(r_base['rouge1'].fmeasure)
        baseline_rls.append(r_base['rougeL'].fmeasure)

    # RAG reader
    context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)]) if contexts else "No reference answers available."
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nQuestion ({lang}): {q}\n\nReference answers:\n{context_str}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700)
    inputs = {k: v.to(reader_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = reader_model.generate(
            **inputs, max_new_tokens=512, temperature=0.3,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    for marker in ['<|im_end|>', '<|endoftext|>', '<|im_start|>']:
        if marker in gen_text:
            gen_text = gen_text.split(marker)[0].strip()
    if gen_text:
        r = scorer.score(ref, gen_text)
        gen_r1s.append(r['rouge1'].fmeasure)
        gen_rls.append(r['rougeL'].fmeasure)

log(f"\n{'='*60}")
log(f"RESULTS ({sample_n} val samples)")
log(f"{'='*60}")
log(f"{'Method':<25} {'ROUGE-1':>10} {'ROUGE-L':>10}")
log(f"{'-'*48}")
if baseline_r1s:
    log(f"{'AfriE5 top-1':<25} {np.mean(baseline_r1s):>10.4f} {np.mean(baseline_rls):>10.4f}")
if gen_r1s:
    log(f"{'Qwen RAG reader':<25} {np.mean(gen_r1s):>10.4f} {np.mean(gen_rls):>10.4f}")
    diff_r1 = np.mean(gen_r1s) - np.mean(baseline_r1s) if baseline_r1s else 0
    diff_rl = np.mean(gen_rls) - np.mean(baseline_rls) if baseline_rls else 0
    log(f"{'Improvement':<25} {diff_r1:>+10.4f} {diff_rl:>+10.4f}")
log(f"{'-'*48}")
log(f"\nPrevious LB best: 0.6545")
log(f"submission_rag_reader.csv saved to Drive!")
log("Done.")
