"""
=============================================================================
OVERNIGHT MEGA-PIPELINE: Cross-Encoder Reranker + Fine-Tuned RAG Reader
=============================================================================
Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm scikit-learn peft trl bitsandbytes accelerate datasets
Cell 3: Paste this entire script
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
log(f"Output: {OUTPUT_DIR} (saved to Google Drive)")

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
subsets_raw   = combined['subset'].tolist()
log(f"Combined corpus: {len(combined)} samples")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Ghana)', 'Amh_Eth': 'Amharic (Ethiopia)',
    'Eng_Eth': 'English (Ethiopia)', 'Eng_Gha': 'English (Ghana)',
    'Eng_Ken': 'English (Kenya)', 'Eng_Uga': 'English (Uganda)',
    'Lug_Uga': 'Luganda (Uganda)', 'Swa_Ken': 'Swahili (Kenya)',
}

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu_name} | {gpu_mem:.1f} GB")

all_results = []

# ============================================================
# PHASE 1: LOAD AfriE5 + BUILD INDEX + ORACLE
# ============================================================
log(f"\n{'='*70}")
log("PHASE 1: Load AfriE5 retriever + Oracle analysis")
log(f"{'='*70}")

from sentence_transformers import SentenceTransformer, InputExample

PREFIX = "query: "

if AFRIE5_DIR and AFRIE5_DIR.exists():
    bienc = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0')
    log(f"AfriE5 loaded from Drive: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")
else:
    bienc = SentenceTransformer('McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
    log(f"Loaded from HuggingFace: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

log("Encoding corpus questions...")
corpus_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
fidx = faiss.IndexFlatIP(corpus_emb.shape[1])
fidx.add(corpus_emb)
log(f"  Index built: {corpus_emb.shape}")

log("Encoding val questions...")
val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

# --- ORACLE ANALYSIS ---
log("\n--- Oracle Analysis ---")
TOP_K = 50
oracle_r1s, oracle_rls = [], []
current_r1s, current_rls = [], []
per_lang = defaultdict(lambda: {'curr_r1': [], 'curr_rl': [], 'orc_r1': [], 'orc_rl': []})

for i in tqdm(range(len(val_df)), desc="Oracle"):
    q   = str(val_df.iloc[i]['input']).strip()
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref:
        continue

    D, I = fidx.search(val_emb[i:i+1], TOP_K)

    curr_answer = ''
    for j in range(TOP_K):
        ci = int(I[0][j])
        if ci >= len(combined):
            continue
        if str(combined.iloc[ci]['input']).strip() != q:
            curr_answer = str(combined.iloc[ci]['output'])
            break
    if not curr_answer:
        curr_answer = str(combined.iloc[int(I[0][0])]['output'])

    rc = scorer.score(ref, curr_answer)
    cr1 = rc['rouge1'].fmeasure
    crl = rc['rougeL'].fmeasure
    current_r1s.append(cr1)
    current_rls.append(crl)

    best_r1 = cr1
    best_rl = crl
    for j in range(TOP_K):
        ci = int(I[0][j])
        if ci >= len(combined):
            continue
        if str(combined.iloc[ci]['input']).strip() == q:
            continue
        ro = scorer.score(ref, str(combined.iloc[ci]['output']))
        s = ro['rouge1'].fmeasure + ro['rougeL'].fmeasure
        if s > best_r1 + best_rl:
            best_r1 = ro['rouge1'].fmeasure
            best_rl = ro['rougeL'].fmeasure

    oracle_r1s.append(best_r1)
    oracle_rls.append(best_rl)
    per_lang[sub]['curr_r1'].append(cr1)
    per_lang[sub]['curr_rl'].append(crl)
    per_lang[sub]['orc_r1'].append(best_r1)
    per_lang[sub]['orc_rl'].append(best_rl)

c_r1 = np.mean(current_r1s)
c_rl = np.mean(current_rls)
o_r1 = np.mean(oracle_r1s)
o_rl = np.mean(oracle_rls)

log(f"\n{'':20} {'ROUGE-1':>10} {'ROUGE-L':>10}")
log(f"{'-'*42}")
log(f"{'Current (top-1)':20} {c_r1:>10.4f} {c_rl:>10.4f}")
log(f"{'Oracle (best@50)':20} {o_r1:>10.4f} {o_rl:>10.4f}")
log(f"{'ROOM TO IMPROVE':20} {o_r1-c_r1:>+10.4f} {o_rl-c_rl:>+10.4f}")

log(f"\nPer-language:")
for sub in sorted(per_lang.keys()):
    d = per_lang[sub]
    log(f"  {sub:<12} R1: {np.mean(d['curr_r1']):.4f} -> {np.mean(d['orc_r1']):.4f} ({np.mean(d['orc_r1'])-np.mean(d['curr_r1']):+.3f})  "
        f"RL: {np.mean(d['curr_rl']):.4f} -> {np.mean(d['orc_rl']):.4f} ({np.mean(d['orc_rl'])-np.mean(d['curr_rl']):+.3f})")

all_results.append(("afrie5_top1", c_r1, c_rl))
rerank_gap = (o_r1 - c_r1) + (o_rl - c_rl)
log(f"\nTotal reranking headroom: {rerank_gap:.4f}")

# ============================================================
# PHASE 2: CROSS-ENCODER RERANKER
# ============================================================
log(f"\n{'='*70}")
log("PHASE 2: Cross-encoder reranker trained on ROUGE")
log(f"{'='*70}")

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
    from torch.utils.data import DataLoader

    log("Building cross-encoder training data...")
    train_qs_list = train_df['input'].fillna('').astype(str).tolist()
    train_as_list = train_df['output'].fillna('').astype(str).tolist()

    train_q_emb = bienc.encode(
        [f"{PREFIX}{q}" for q in train_qs_list],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    ce_pairs = []
    CE_K = 15

    for i in tqdm(range(len(train_df)), desc="CE train data"):
        q = train_qs_list[i]
        ref = train_as_list[i]
        if not q.strip() or not ref.strip():
            continue

        D, I = fidx.search(train_q_emb[i:i+1], CE_K + 5)
        count = 0
        for j in range(CE_K + 5):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q.strip():
                continue
            ca = str(combined.iloc[ci]['output'])
            r = scorer.score(ref, ca)
            label = 0.5 * r['rouge1'].fmeasure + 0.5 * r['rougeL'].fmeasure
            ce_pairs.append((q, ca, label))
            count += 1
            if count >= CE_K:
                break

    log(f"CE training pairs: {len(ce_pairs)}")

    ce_val_pairs = []
    for i in tqdm(range(min(2000, len(val_df))), desc="CE val data"):
        q = val_qs[i]
        ref = str(val_df.iloc[i]['output']).strip()
        if not q.strip() or not ref.strip():
            continue
        D, I = fidx.search(val_emb[i:i+1], CE_K)
        for j in range(CE_K):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q.strip():
                continue
            ca = str(combined.iloc[ci]['output'])
            r = scorer.score(ref, ca)
            label = 0.5 * r['rouge1'].fmeasure + 0.5 * r['rougeL'].fmeasure
            ce_val_pairs.append((q, ca, label))

    log(f"CE val pairs: {len(ce_val_pairs)}")

    bienc.cpu()
    del train_q_emb
    gc.collect()
    torch.cuda.empty_cache()

    CE_MODEL = "xlm-roberta-base"
    ce_model = CrossEncoder(CE_MODEL, num_labels=1, max_length=256, device='cuda:0')
    log(f"Cross-encoder: {CE_MODEL}")

    ce_train_examples = [InputExample(texts=[q, a], label=l) for q, a, l in ce_pairs]
    ce_val_examples   = [InputExample(texts=[q, a], label=l) for q, a, l in ce_val_pairs]

    ce_loader = DataLoader(ce_train_examples, shuffle=True, batch_size=64)
    ce_evaluator = CECorrelationEvaluator.from_input_examples(ce_val_examples[:5000], name="val")

    CE_EPOCHS = 3
    CE_WARMUP = int(len(ce_loader) * CE_EPOCHS * 0.1)
    log(f"Training: {len(ce_train_examples)} examples, {CE_EPOCHS} epochs, warmup={CE_WARMUP}")

    ce_model.fit(
        train_dataloader=ce_loader,
        evaluator=ce_evaluator,
        evaluation_steps=5000,
        epochs=CE_EPOCHS,
        warmup_steps=CE_WARMUP,
        output_path=str(OUTPUT_DIR / 'ce-rouge-predictor'),
        show_progress_bar=True,
        use_amp=True,
    )
    log("Cross-encoder training complete!")
    ce_model.save(str(OUTPUT_DIR / 'ce-rouge-predictor'))

    del ce_pairs, ce_val_pairs, ce_train_examples, ce_val_examples, ce_loader
    gc.collect()

    # --- EVALUATE RERANKING ON VAL ---
    log("\nEvaluating reranking on val...")
    reranked_r1s = []
    reranked_rls = []
    RERANK_K = 30

    for i in tqdm(range(len(val_df)), desc="Reranking val"):
        q = str(val_df.iloc[i]['input']).strip()
        ref = str(val_df.iloc[i]['output']).strip()
        if not ref:
            continue

        D, I = fidx.search(val_emb[i:i+1], RERANK_K + 5)
        cands = []
        for j in range(RERANK_K + 5):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q:
                continue
            cands.append(str(combined.iloc[ci]['output']))
            if len(cands) >= RERANK_K:
                break

        if not cands:
            reranked_r1s.append(0)
            reranked_rls.append(0)
            continue

        ce_scores = ce_model.predict([(q, c) for c in cands], show_progress_bar=False)
        best = cands[int(np.argmax(ce_scores))]
        r = scorer.score(ref, best)
        reranked_r1s.append(r['rouge1'].fmeasure)
        reranked_rls.append(r['rougeL'].fmeasure)

    rr1 = np.mean(reranked_r1s)
    rrl = np.mean(reranked_rls)
    log(f"\n{'':20} {'ROUGE-1':>10} {'ROUGE-L':>10}")
    log(f"{'Before (top-1)':20} {c_r1:>10.4f} {c_rl:>10.4f}")
    log(f"{'After (CE rerank)':20} {rr1:>10.4f} {rrl:>10.4f}")
    log(f"{'IMPROVEMENT':20} {rr1-c_r1:>+10.4f} {rrl-c_rl:>+10.4f}")
    log(f"{'Oracle ceiling':20} {o_r1:>10.4f} {o_rl:>10.4f}")
    all_results.append(("ce_reranked", rr1, rrl))

    # --- GENERATE CE SUBMISSION ---
    log("\nGenerating CE reranked test submission...")
    bienc.to('cuda:0')
    test_inputs = test_df['input'].fillna('').astype(str).tolist()
    test_emb = bienc.encode(
        [f"{PREFIX}{q}" for q in test_inputs],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    bienc.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for i in tqdm(range(len(test_df)), desc="CE test submission"):
        q = str(test_df.iloc[i]['input']).strip()
        D, I = fidx.search(test_emb[i:i+1], RERANK_K + 5)
        cands = []
        for j in range(RERANK_K + 5):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q:
                continue
            cands.append(str(combined.iloc[ci]['output']))
            if len(cands) >= RERANK_K:
                break

        if cands:
            ce_scores = ce_model.predict([(q, c) for c in cands], show_progress_bar=False)
            answer = cands[int(np.argmax(ce_scores))]
        else:
            answer = "No answer found."

        rows.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': answer,
            'TargetR1F1': answer,
            'TargetLLM': answer,
        })

    sub_ce = pd.DataFrame(rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    assert len(sub_ce) == len(sample_sub), f"CE submission length mismatch: {len(sub_ce)} vs {len(sample_sub)}"
    sub_ce.to_csv(OUTPUT_DIR / 'submission_ce_reranked.csv', index=False)
    log("Saved: submission_ce_reranked.csv")

    del ce_model
    gc.collect()
    torch.cuda.empty_cache()

except Exception as e:
    log(f"PHASE 2 FAILED: {e}")
    traceback.print_exc()
    log("Continuing to Phase 3...")
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# PHASE 3: FINE-TUNED RAG READER (Qwen2.5-7B with LoRA)
# ============================================================
log(f"\n{'='*70}")
log("PHASE 3: Fine-tune Qwen2.5-7B-Instruct as RAG reader")
log(f"{'='*70}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    READER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    log(f"Loading {READER_MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - try flash attention, fall back gracefully
    try:
        reader_model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        log("Using Flash Attention 2")
    except Exception:
        reader_model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        log("Using default attention")

    log(f"Model loaded: {sum(p.numel() for p in reader_model.parameters())/1e9:.1f}B params")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- BUILD TRAINING DATA ---
    log("\nBuilding RAG training data...")
    bienc.to('cuda:0')
    train_qs_text = train_df['input'].fillna('').astype(str).tolist()
    train_as_text = train_df['output'].fillna('').astype(str).tolist()
    train_subsets = train_df['subset'].fillna('').astype(str).tolist()

    train_q_emb = bienc.encode(
        [f"{PREFIX}{q}" for q in train_qs_text],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    bienc.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    SYSTEM_PROMPT = (
        "You are a multilingual health QA assistant. "
        "Answer the question using the provided reference answers. "
        "Keep the exact same wording, phrasing, and medical terminology as the references. "
        "Do NOT add information not in the references. "
        "Answer in the same language as the question."
    )

    train_texts = []
    for i in tqdm(range(len(train_df)), desc="Building RAG data"):
        q = train_qs_text[i]
        ref_answer = train_as_text[i]
        lang = SUBSET_TO_LANG.get(train_subsets[i], train_subsets[i])
        if not q.strip() or not ref_answer.strip():
            continue

        D, I = fidx.search(train_q_emb[i:i+1], 10)
        contexts = []
        for j in range(10):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            cq = str(combined.iloc[ci]['input']).strip()
            ca = str(combined.iloc[ci]['output']).strip()
            if cq == q.strip():
                continue
            if ca == ref_answer.strip():
                continue
            contexts.append(ca)
            if len(contexts) >= 3:
                break

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
    del train_q_emb
    gc.collect()

    val_texts = []
    for i in range(min(1000, len(val_df))):
        q = val_qs[i]
        ref_answer = str(val_df.iloc[i]['output']).strip()
        lang = SUBSET_TO_LANG.get(str(val_df.iloc[i]['subset']), str(val_df.iloc[i]['subset']))
        if not q.strip() or not ref_answer.strip():
            continue

        D, I = fidx.search(val_emb[i:i+1], 10)
        contexts = []
        for j in range(10):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q.strip():
                continue
            ca = str(combined.iloc[ci]['output']).strip()
            if ca == ref_answer.strip():
                continue
            contexts.append(ca)
            if len(contexts) >= 3:
                break

        if contexts:
            context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)])
        else:
            context_str = "No reference answers available."

        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Question ({lang}): {q}\n\n"
            f"Reference answers:\n{context_str}<|im_end|>\n"
            f"<|im_start|>assistant\n{ref_answer}<|im_end|>"
        )
        val_texts.append(text)

    log(f"Val examples: {len(val_texts)}")

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset   = Dataset.from_dict({"text": val_texts})

    # --- TRAIN ---
    log("\nStarting LoRA fine-tuning...")
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / 'qwen-rag-reader'),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
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
        max_seq_length=768,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        save_total_limit=1,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=reader_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    log("LoRA fine-tuning complete!")

    trainer.save_model(str(OUTPUT_DIR / 'qwen-rag-lora'))
    tokenizer.save_pretrained(str(OUTPUT_DIR / 'qwen-rag-lora'))
    log("LoRA adapter saved to Drive!")

    reader_model = trainer.model
    reader_model.eval()

    # --- GENERATE TEST ANSWERS ---
    log("\nGenerating test answers with fine-tuned reader...")
    bienc.to('cuda:0')
    test_inputs_list = test_df['input'].fillna('').astype(str).tolist()
    test_subsets_list = test_df['subset'].fillna('').astype(str).tolist()
    test_emb_gen = bienc.encode(
        [f"{PREFIX}{q}" for q in test_inputs_list],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    bienc.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    gen_rows = []
    for i in tqdm(range(len(test_df)), desc="Generating answers"):
        q = test_inputs_list[i]
        lang = SUBSET_TO_LANG.get(test_subsets_list[i], test_subsets_list[i])

        D, I = fidx.search(test_emb_gen[i:i+1], 10)
        contexts = []
        for j in range(10):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q.strip():
                continue
            contexts.append(str(combined.iloc[ci]['output']))
            if len(contexts) >= 3:
                break

        if contexts:
            context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)])
        else:
            context_str = "No reference answers available."

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
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

        # Clean up Qwen chat markers
        for marker in ['<|im_end|>', '<|endoftext|>', '<|im_start|>']:
            if marker in gen_text:
                gen_text = gen_text.split(marker)[0].strip()

        if not gen_text:
            gen_text = "No answer available."

        gen_rows.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': gen_text,
            'TargetR1F1': gen_text,
            'TargetLLM': gen_text,
        })

        if (i + 1) % 500 == 0:
            log(f"  Generated {i+1}/{len(test_df)}...")
            tmp = pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
            tmp.to_csv(OUTPUT_DIR / 'submission_rag_reader_PARTIAL.csv', index=False)

    sub_gen = pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    assert len(sub_gen) == len(sample_sub), f"RAG submission length mismatch: {len(sub_gen)} vs {len(sample_sub)}"

    for col in ['TargetRLF1', 'TargetR1F1', 'TargetLLM']:
        sub_gen[col] = sub_gen[col].fillna("No answer available.")
        sub_gen[col] = sub_gen[col].replace('', "No answer available.")

    sub_gen.to_csv(OUTPUT_DIR / 'submission_rag_reader.csv', index=False)
    log("Saved: submission_rag_reader.csv")

    # --- QUICK VAL EVAL ---
    log("\nEvaluating RAG reader on val (200 samples)...")
    gen_r1s = []
    gen_rls = []
    sample_n = min(200, len(val_df))

    for i in tqdm(range(sample_n), desc="Eval RAG"):
        q = val_qs[i]
        ref = str(val_df.iloc[i]['output']).strip()
        lang = SUBSET_TO_LANG.get(str(val_df.iloc[i]['subset']), str(val_df.iloc[i]['subset']))
        if not ref:
            continue

        D, I = fidx.search(val_emb[i:i+1], 10)
        contexts = []
        for j in range(10):
            ci = int(I[0][j])
            if ci >= len(combined):
                continue
            if str(combined.iloc[ci]['input']).strip() == q.strip():
                continue
            contexts.append(str(combined.iloc[ci]['output']))
            if len(contexts) >= 3:
                break

        if contexts:
            context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)])
        else:
            context_str = "No reference answers available."

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nQuestion ({lang}): {q}\n\nReference answers:\n{context_str}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700)
        inputs = {k: v.to(reader_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = reader_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
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

    if gen_r1s:
        gr1 = np.mean(gen_r1s)
        grl = np.mean(gen_rls)
        log(f"RAG Reader val: R1={gr1:.4f} RL={grl:.4f} (on {len(gen_r1s)} samples)")
        all_results.append(("rag_reader", gr1, grl))

    del reader_model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

except Exception as e:
    log(f"PHASE 3 FAILED: {e}")
    traceback.print_exc()
    log("RAG reader submission NOT generated.")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*70}")
log("OVERNIGHT RUN COMPLETE")
log(f"{'='*70}")
log(f"\n{'Strategy':<25} {'ROUGE-1':>10} {'ROUGE-L':>10}")
log(f"{'-'*48}")
best_r1 = max(r[1] for r in all_results) if all_results else 0
for name, r1v, rlv in sorted(all_results, key=lambda x: x[1], reverse=True):
    marker = " <-- BEST" if r1v == best_r1 else ""
    log(f"{name:<25} {r1v:>10.4f} {rlv:>10.4f}{marker}")
log(f"{'-'*48}")
log(f"Previous LB best: 0.6545")
log(f"Oracle ceiling: R1={o_r1:.4f} RL={o_rl:.4f}")
log(f"\nSUBMISSIONS on Google Drive (multilingual-health-qa/outputs/):")
for f in sorted(OUTPUT_DIR.glob("submission_*.csv")):
    log(f"  -> {f.name}")
log("\nSubmit the highest val-scoring file first!")
