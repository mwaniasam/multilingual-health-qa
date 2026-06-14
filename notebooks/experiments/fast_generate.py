"""
FAST GENERATION — paste after interrupting the slow generation cell.
Model (reader_model) and all variables are still in memory.
Key speedups: batched inference, greedy decoding, shorter max tokens.
"""
log("Fast generation starting...")

# Set left-padding for batched generation
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

reader_model.eval()

# Build ALL prompts first
log("Building prompts...")
all_prompts = []
for i in range(len(test_df)):
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
    all_prompts.append(prompt)

log(f"Built {len(all_prompts)} prompts")

# Batched generation — 4x faster
BATCH_SIZE = 4
MAX_NEW_TOKENS = 256  # most answers are < 200 tokens
gen_rows = []

for batch_start in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Generating"):
    batch_end = min(batch_start + BATCH_SIZE, len(test_df))
    batch_prompts = all_prompts[batch_start:batch_end]

    inputs = tokenizer(
        batch_prompts, return_tensors="pt", truncation=True,
        max_length=700, padding=True,
    )
    inputs = {k: v.to(reader_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = reader_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy = faster
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    for idx_in_batch in range(batch_end - batch_start):
        global_idx = batch_start + idx_in_batch
        prompt_len = inputs['input_ids'][idx_in_batch].ne(tokenizer.pad_token_id).sum().item()
        gen_ids = outputs[idx_in_batch][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

        for marker in ['<|im_end|>', '<|endoftext|>', '<|im_start|>']:
            if marker in gen_text:
                gen_text = gen_text.split(marker)[0].strip()

        if not gen_text:
            gen_text = "No answer available."

        gen_rows.append({
            'ID': test_df.iloc[global_idx]['ID'],
            'TargetRLF1': gen_text,
            'TargetR1F1': gen_text,
            'TargetLLM': gen_text,
        })

    if (batch_end) % 500 < BATCH_SIZE:
        log(f"  Progress: {batch_end}/{len(test_df)}")
        pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']].to_csv(
            OUTPUT_DIR / 'submission_rag_reader_PARTIAL.csv', index=False)

# Save final
sub_gen = pd.DataFrame(gen_rows)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_gen) == len(sample_sub), f"Length: {len(sub_gen)} vs {len(sample_sub)}"
for col in ['TargetRLF1', 'TargetR1F1', 'TargetLLM']:
    sub_gen[col] = sub_gen[col].fillna("No answer available.")
    sub_gen[col] = sub_gen[col].replace('', "No answer available.")
sub_gen.to_csv(OUTPUT_DIR / 'submission_rag_reader.csv', index=False)
log("Saved: submission_rag_reader.csv")

# Quick val eval
log("\nEvaluating on val (200 samples)...")
gen_r1s, gen_rls = [], []
baseline_r1s, baseline_rls = [], []

for i in tqdm(range(min(200, len(val_df))), desc="Val eval"):
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
        if not baseline_answer: baseline_answer = ca
        contexts.append(ca)
        if len(contexts) >= 3: break

    if baseline_answer:
        r_base = scorer.score(ref, baseline_answer)
        baseline_r1s.append(r_base['rouge1'].fmeasure)
        baseline_rls.append(r_base['rougeL'].fmeasure)

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
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
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

log(f"\n{'='*50}")
log(f"{'Method':<25} {'ROUGE-1':>10} {'ROUGE-L':>10}")
log(f"{'-'*48}")
if baseline_r1s:
    log(f"{'AfriE5 top-1':<25} {np.mean(baseline_r1s):>10.4f} {np.mean(baseline_rls):>10.4f}")
if gen_r1s:
    log(f"{'Qwen RAG reader':<25} {np.mean(gen_r1s):>10.4f} {np.mean(gen_rls):>10.4f}")
log(f"{'='*50}")
log("Done! submission_rag_reader.csv on Drive.")
