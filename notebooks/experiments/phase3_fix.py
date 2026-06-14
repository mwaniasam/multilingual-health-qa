"""
PHASE 3 FIX — Paste as new Colab cell after the main script finishes.
All variables (bienc, fidx, combined, val_emb, etc.) are still in memory.
"""
import inspect
import trl
log(f"trl version: {trl.__version__}")

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset

# Check if SFTConfig exists and what it accepts
try:
    from trl import SFTConfig
    sft_params = list(inspect.signature(SFTConfig.__init__).parameters.keys())
    log(f"SFTConfig params (first 20): {sft_params[:20]}")
    HAS_SFT_CONFIG = True
except ImportError:
    HAS_SFT_CONFIG = False
    log("SFTConfig not available, using TrainingArguments")

# Check what SFTTrainer accepts
sft_trainer_params = list(inspect.signature(SFTTrainer.__init__).parameters.keys())
log(f"SFTTrainer params: {sft_trainer_params}")

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
except Exception:
    reader_model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    log("Using default attention")

log(f"Model loaded: {sum(p.numel() for p in reader_model.parameters())/1e9:.1f}B params")

lora_config = LoraConfig(
    r=64, lora_alpha=128, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

# --- BUILD TRAINING DATA (reuses bienc, fidx from main script) ---
log("\nBuilding RAG training data...")
bienc.to('cuda:0')
train_qs_text = train_df['input'].fillna('').astype(str).tolist()
train_as_text = train_df['output'].fillna('').astype(str).tolist()
train_subsets = train_df['subset'].fillna('').astype(str).tolist()

train_q_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in train_qs_text],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
bienc.cpu(); gc.collect(); torch.cuda.empty_cache()

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
del train_q_emb; gc.collect()

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

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset   = Dataset.from_dict({"text": val_texts})

# --- TRAIN (version-robust) ---
log("\nStarting LoRA fine-tuning...")

# Use plain TrainingArguments (works with ALL trl versions)
training_args = TrainingArguments(
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
    report_to="none",
    save_total_limit=1,
    dataloader_num_workers=2,
)

# Build SFTTrainer kwargs — only include params that the installed version accepts
sft_kwargs = {
    "model": reader_model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    "peft_config": lora_config,
}

# Add optional params based on what SFTTrainer accepts
optional_params = {
    "tokenizer": tokenizer,
    "processing_class": tokenizer,
    "max_seq_length": 768,
    "dataset_text_field": "text",
    "packing": False,
}

for param, value in optional_params.items():
    if param in sft_trainer_params:
        sft_kwargs[param] = value
        log(f"  Using SFTTrainer param: {param}")

# Don't pass both tokenizer and processing_class
if "processing_class" in sft_kwargs and "tokenizer" in sft_kwargs:
    del sft_kwargs["tokenizer"]
    log("  Removed duplicate 'tokenizer' (using 'processing_class')")

trainer = SFTTrainer(**sft_kwargs)
trainer.train()
log("LoRA fine-tuning complete!")

trainer.save_model(str(OUTPUT_DIR / 'qwen-rag-lora'))
tokenizer.save_pretrained(str(OUTPUT_DIR / 'qwen-rag-lora'))
log("LoRA adapter saved to Drive!")

reader_model = trainer.model
reader_model.eval()

# --- GENERATE TEST ANSWERS ---
log("\nGenerating test answers...")
bienc.to('cuda:0')
test_inputs_list = test_df['input'].fillna('').astype(str).tolist()
test_subsets_list = test_df['subset'].fillna('').astype(str).tolist()
test_emb_gen = bienc.encode(
    [f"{PREFIX}{q}" for q in test_inputs_list],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)
bienc.cpu(); gc.collect(); torch.cuda.empty_cache()

gen_rows = []
for i in tqdm(range(len(test_df)), desc="Generating"):
    q = test_inputs_list[i]
    lang = SUBSET_TO_LANG.get(test_subsets_list[i], test_subsets_list[i])

    D, I = fidx.search(test_emb_gen[i:i+1], 10)
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
        log(f"  Generated {i+1}/{len(test_df)}...")
        tmp = pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
        tmp.to_csv(OUTPUT_DIR / 'submission_rag_reader_PARTIAL.csv', index=False)

sub_gen = pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_gen) == len(sample_sub)
for col in ['TargetRLF1', 'TargetR1F1', 'TargetLLM']:
    sub_gen[col] = sub_gen[col].fillna("No answer available.")
    sub_gen[col] = sub_gen[col].replace('', "No answer available.")
sub_gen.to_csv(OUTPUT_DIR / 'submission_rag_reader.csv', index=False)
log("Saved: submission_rag_reader.csv")

# --- QUICK VAL EVAL ---
log("\nEvaluating on val (200 samples)...")
gen_r1s, gen_rls = [], []
for i in tqdm(range(min(200, len(val_df))), desc="Eval"):
    q = val_qs[i]
    ref = str(val_df.iloc[i]['output']).strip()
    lang = SUBSET_TO_LANG.get(str(val_df.iloc[i]['subset']), str(val_df.iloc[i]['subset']))
    if not ref: continue

    D, I = fidx.search(val_emb[i:i+1], 10)
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

if gen_r1s:
    log(f"\n{'='*50}")
    log(f"RAG Reader val: R1={np.mean(gen_r1s):.4f} RL={np.mean(gen_rls):.4f}")
    log(f"AfriE5 top-1:   R1={c_r1:.4f} RL={c_rl:.4f}")
    log(f"Oracle ceiling:  R1={o_r1:.4f} RL={o_rl:.4f}")
    log(f"{'='*50}")

log("\nDone! submission_rag_reader.csv saved to Drive.")
