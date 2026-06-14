"""
=============================================================================
CELL 5: GENERATE ANSWERS + EVALUATE (~30-40 min on A100)
=============================================================================
Generates answers for val (evaluate per-language) and test (submit).
Per-language gating: only use generated answers where they beat retrieval.
=============================================================================
"""
import torch, gc, json, time
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

FT_MODEL_DIR = OUTPUT_DIR / 'qwen-ft-health'
GEN_CACHE = OUTPUT_DIR / 'gen_cache'
GEN_CACHE.mkdir(parents=True, exist_ok=True)

# ---- Load fine-tuned model ----
log("Loading fine-tuned model...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(str(FT_MODEL_DIR), trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, str(FT_MODEL_DIR))
model.eval()
log(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

# ---- Generation function ----
def generate_answer(question, contexts, language, max_new_tokens=512):
    """Generate an answer using the fine-tuned model."""
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Extract only the generated part
    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Remove any trailing artifacts
    for stop in ['<|im_end|>', '<|im_start|>', '<|endoftext|>']:
        if stop in answer:
            answer = answer[:answer.index(stop)].strip()

    return answer

def get_context(q_text, q_emb, subset, k=3):
    """Get top-k retrieved answers as context."""
    qs = q_text.strip()
    if subset in lang_indices:
        idx, mask = lang_indices[subset]
        D, I = idx.search(np.asarray(q_emb, np.float32).reshape(1, -1), k + 3)
        out = []
        for li in I[0]:
            if li < 0: continue
            ci = mask[int(li)]
            if corpus_q_stripped[ci] == qs: continue
            out.append(answers_raw[ci])
            if len(out) >= k: break
        return out
    return []

# ==========================================================================
# PART A: GENERATE + EVALUATE ON VAL (per-language comparison)
# ==========================================================================
log(f"\n{'='*60}")
log("PART A: Val evaluation (per-language)")
log(f"{'='*60}")

# Use a sample of val for speed (400 samples, stratified)
VAL_SAMPLE_N = 400
val_sample_indices = []
per_lang_count = defaultdict(int)
target_per_lang = VAL_SAMPLE_N // len(SUBSET_TO_LANG)

for sub in sorted(SUBSET_TO_LANG.keys()):
    sub_idx = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset']) == sub]
    np.random.seed(42)
    sample = np.random.choice(sub_idx, min(target_per_lang, len(sub_idx)), replace=False)
    val_sample_indices.extend(sample.tolist())

log(f"Val sample: {len(val_sample_indices)} questions")

# Check for cached val generations
val_gen_cache = GEN_CACHE / 'val_generated.json'
val_gen = json.load(open(val_gen_cache)) if val_gen_cache.exists() else {}
log(f"Cached val generations: {len(val_gen)}")

# Generate for val sample
for idx_num, i in enumerate(tqdm(val_sample_indices, desc="Generating val")):
    key = str(i)
    if key in val_gen:
        continue

    q = val_qs[i]
    sub = str(val_df.iloc[i]['subset'])
    lang = SUBSET_TO_LANG.get(sub, sub)
    ctx = get_context(q, val_emb[i], sub, k=3)

    if not ctx:
        val_gen[key] = ""
        continue

    try:
        ans = generate_answer(q, ctx, lang)
        val_gen[key] = ans
    except Exception as e:
        log(f"  Error at {i}: {e}")
        val_gen[key] = ctx[0] if ctx else ""

    # Save periodically
    if (idx_num + 1) % 50 == 0:
        json.dump(val_gen, open(val_gen_cache, 'w'))
        log(f"  Saved {len(val_gen)} val generations")

json.dump(val_gen, open(val_gen_cache, 'w'))
log(f"Val generation complete: {len(val_gen)} answers")

# ---- Evaluate ----
log("\nEvaluating generated vs retrieval on val...")

per_lang_ret = defaultdict(lambda: {'r1': [], 'rl': []})
per_lang_gen = defaultdict(lambda: {'r1': [], 'rl': []})

for i in val_sample_indices:
    key = str(i)
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref: continue

    rt = uni_toks(ref)

    # Retrieval baseline (top-1)
    cands = val_cands_all.get(i, get_same_lang_candidates(val_qs[i], val_emb[i], sub))
    if cands:
        ret_ans = cands[0]['answer']
        per_lang_ret[sub]['r1'].append(uni_r1(rt, uni_toks(ret_ans)))
        per_lang_ret[sub]['rl'].append(uni_rl(rt, uni_toks(ret_ans)))

    # Generated
    gen_ans = val_gen.get(key, '')
    if gen_ans:
        per_lang_gen[sub]['r1'].append(uni_r1(rt, uni_toks(gen_ans)))
        per_lang_gen[sub]['rl'].append(uni_rl(rt, uni_toks(gen_ans)))

log(f"\n{'Sub':<12} {'Ret R1':>8} {'Gen R1':>8} {'Δ R1':>7} "
    f"{'Ret RL':>8} {'Gen RL':>8} {'Δ RL':>7}")
log('-' * 70)

gen_wins_r1 = {}  # per lang: does generation beat retrieval for R1?
gen_wins_rl = {}
gen_wins_llm = {}  # assume gen is better for LLM (more fluent)

for sub in sorted(SUBSET_TO_LANG.keys()):
    rr1 = np.mean(per_lang_ret[sub]['r1']) if per_lang_ret[sub]['r1'] else 0
    rrl = np.mean(per_lang_ret[sub]['rl']) if per_lang_ret[sub]['rl'] else 0
    gr1 = np.mean(per_lang_gen[sub]['r1']) if per_lang_gen[sub]['r1'] else 0
    grl = np.mean(per_lang_gen[sub]['rl']) if per_lang_gen[sub]['rl'] else 0

    dr1 = gr1 - rr1
    drl = grl - rrl
    gen_wins_r1[sub] = dr1 > 0.005
    gen_wins_rl[sub] = drl > 0.005
    gen_wins_llm[sub] = True  # assume yes for LLM (fine-tuned should be better)

    marker_r1 = " ★" if gen_wins_r1[sub] else ""
    marker_rl = " ★" if gen_wins_rl[sub] else ""
    log(f"  {sub:<12} {rr1:>8.4f} {gr1:>8.4f} {dr1:>+7.4f}{marker_r1} "
        f"{rrl:>8.4f} {grl:>8.4f} {drl:>+7.4f}{marker_rl}")

log(f"\nPer-language gating decisions:")
for sub in sorted(SUBSET_TO_LANG.keys()):
    log(f"  {sub}: R1={'GEN' if gen_wins_r1[sub] else 'RET'} | "
        f"RL={'GEN' if gen_wins_rl[sub] else 'RET'} | LLM=GEN")

# Save decisions
gating = {sub: {'r1': gen_wins_r1[sub], 'rl': gen_wins_rl[sub], 'llm': True}
          for sub in SUBSET_TO_LANG}
json.dump(gating, open(GEN_CACHE / 'gating_decisions.json', 'w'), indent=2)

# ==========================================================================
# PART B: GENERATE TEST ANSWERS
# ==========================================================================
log(f"\n{'='*60}")
log("PART B: Generate test answers")
log(f"{'='*60}")

test_gen_cache = GEN_CACHE / 'test_generated.json'
test_gen = json.load(open(test_gen_cache)) if test_gen_cache.exists() else {}
log(f"Cached test generations: {len(test_gen)}")

for i in tqdm(range(len(test_df)), desc="Generating test"):
    rid = str(test_df.iloc[i]['ID'])
    if rid in test_gen:
        continue

    q = test_qs[i]
    sub = test_subs[i]
    lang = SUBSET_TO_LANG.get(sub, sub)
    ctx = get_context(q, test_emb[i], sub, k=3)

    if not ctx:
        test_gen[rid] = "No answer."
        continue

    try:
        ans = generate_answer(q, ctx, lang)
        test_gen[rid] = ans
    except Exception as e:
        log(f"  Error at {rid}: {e}")
        test_gen[rid] = ctx[0] if ctx else "No answer."

    if (i + 1) % 100 == 0:
        json.dump(test_gen, open(test_gen_cache, 'w'))
        log(f"  Saved {len(test_gen)} test generations")

json.dump(test_gen, open(test_gen_cache, 'w'))
log(f"Test generation complete: {len(test_gen)} answers")

# Free GPU
del model, base_model
gc.collect()
torch.cuda.empty_cache()
log("GPU freed.")
