"""
=============================================================================
CELL 3: PREPARE TRAINING DATA (paste after bootstrap, ~10 min CPU)
=============================================================================
Retrieves top-3 context answers for each training sample, formats as
chat messages for QLoRA fine-tuning.
=============================================================================
"""
import torch, gc, json, time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ---- Retrieve context for each training sample ----
log("Retrieving top-3 context for each training sample...")
TRAIN_N = len(train_df)
train_contexts = []  # list of list of answer strings

# Training samples are first TRAIN_N rows of combined
for i in tqdm(range(TRAIN_N), desc="Retrieving train context"):
    q = corpus_q_stripped[i]
    sub = subsets_raw[i]

    # Search same-language index, exclude exact match
    if sub in lang_indices:
        idx, mask = lang_indices[sub]
        D, I = idx.search(corpus_emb[i:i+1], 8)
        ctx = []
        for li in I[0]:
            if li < 0: continue
            ci = mask[int(li)]
            if ci == i: continue  # exclude self
            if corpus_q_stripped[ci] == q: continue  # exclude exact text match
            ctx.append(answers_raw[ci])
            if len(ctx) >= 3: break
        train_contexts.append(ctx)
    else:
        train_contexts.append([])

log(f"Retrieved context for {len(train_contexts)} training samples")
log(f"Samples with 3 contexts: {sum(1 for c in train_contexts if len(c) >= 3)}")
log(f"Samples with 0 contexts: {sum(1 for c in train_contexts if len(c) == 0)}")

# ---- Format training data ----
log("Formatting training data...")

def format_chat(question, ref_answer, contexts, language):
    """Format as Qwen2.5 ChatML template."""
    ctx_str = "\n".join([f"{i+1}. {a[:500]}" for i, a in enumerate(contexts[:3])])

    system = (
        f"You are a multilingual health expert. Answer health questions based on "
        f"the reference information provided. Use the EXACT words and phrases from "
        f"the references when possible. Be complete and accurate. Answer in {language}."
    )
    user = (
        f"Question: {question}\n\n"
        f"Reference answers:\n{ctx_str}\n\n"
        f"Provide a comprehensive answer in {language}:"
    )
    assistant = ref_answer[:800]  # truncate very long answers

    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return text

train_texts = []
for i in range(TRAIN_N):
    q = questions_raw[i]
    ref = answers_raw[i]
    sub = subsets_raw[i]
    ctx = train_contexts[i]
    lang = SUBSET_TO_LANG.get(sub, sub)

    if not q.strip() or not ref.strip():
        continue
    if len(ctx) < 1:
        # Use ref itself as context (self-training signal)
        ctx = [ref]

    text = format_chat(q, ref, ctx, lang)
    train_texts.append(text)

log(f"Training samples formatted: {len(train_texts)}")

# Save to Drive for persistence
train_data_path = OUTPUT_DIR / 'ft_train_data.json'
json.dump(train_texts, open(train_data_path, 'w'))
log(f"Saved training data to {train_data_path}")

# Quick stats
lengths = [len(t.split()) for t in train_texts]
log(f"Token stats: min={min(lengths)}, median={np.median(lengths):.0f}, "
    f"max={max(lengths)}, mean={np.mean(lengths):.0f}")
