"""
=============================================================================
COMPLETE NOTEBOOK — 3 CELLS ONLY
=============================================================================
Cell 1: !pip install -q transformers peft bitsandbytes accelerate faiss-cpu rouge-score tqdm pylcs
Cell 2: Paste your bootstrap (the one that loads all caches from Drive)
Cell 3: Paste THIS file
=============================================================================
"""

# ===========================================================================
# STEP 1: LOAD FINE-TUNED MODEL (4-BIT — 3.5GB, FAST)
# ===========================================================================
import torch, gc, json, re as _re2, time
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

FT_MODEL_DIR = OUTPUT_DIR / 'qwen-ft-health'
assert (FT_MODEL_DIR / 'adapter_config.json').exists(), \
    f"No fine-tuned model at {FT_MODEL_DIR}. Train first."

log("Loading fine-tuned Qwen in 4-bit (fast inference)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(str(FT_MODEL_DIR))
base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    quantization_config=bnb,
    device_map='auto',
    trust_remote_code=True,
)
ft = PeftModel.from_pretrained(base, str(FT_MODEL_DIR))
ft.eval()
log(f"Model loaded in 4-bit. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ===========================================================================
# FAST GENERATION FUNCTIONS (short context, low max_tokens)
# ===========================================================================
@torch.no_grad()
def ft_generate(q, lang, cands, max_new=200):
    # Only top-3 context, truncated to 200 chars each
    ctx = "\n".join(f"{k+1}. {c['answer'][:200]}" for k, c in enumerate(cands[:3]))
    msgs = [
        {"role": "system", "content":
         f"You are a health expert. Answer using the EXACT words from the references. "
         f"Be complete. Answer in {lang}."},
        {"role": "user", "content":
         f"Question: {q}\n\nReferences:\n{ctx}\n\nAnswer in {lang}:"}
    ]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=1024).to(ft.device)
    out = ft.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    ans = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    for stop in ['<|im_end|>', '<|im_start|>', '<|endoftext|>']:
        if stop in ans: ans = ans[:ans.index(stop)].strip()
    return ans

# Speed test
log("Speed test...")
test_cands = val_cands_all.get(0, get_same_lang_candidates(val_qs[0], val_emb[0],
              str(val_df.iloc[0]['subset'])))
t0 = time.time()
test_ans = ft_generate(val_qs[0], 'English', test_cands)
log(f"Speed: {time.time()-t0:.1f}s per query. Answer: {test_ans[:100]}...")

# ===========================================================================
# STEP 2: VAL GENERATION (400 samples, ~15-30 min)
# ===========================================================================
log(f"\n{'='*60}")
log("STEP 2: Val generation (400 samples)")
log(f"{'='*60}")

rng = np.random.default_rng(42)
val_sample = []
for sub in SUBSET_TO_LANG:
    idxs = [i for i in range(len(val_df))
            if str(val_df.iloc[i]['subset']) == sub and i in val_cands_all and val_cands_all[i]]
    val_sample += [int(x) for x in rng.choice(idxs, min(50, len(idxs)), replace=False)]
log(f"Val sample: {len(val_sample)} queries")

fprog = OUTPUT_DIR / 'ftqwen_val_sample.json'
fans = json.load(open(fprog)) if fprog.exists() else {}
log(f"Already generated: {len(fans)}")

t0 = time.time()
for n, i in enumerate(tqdm(val_sample, desc="FT-Qwen val")):
    if str(i) in fans:
        continue
    sub = str(val_df.iloc[i]['subset'])
    cands = val_cands_all.get(i, get_same_lang_candidates(val_qs[i], val_emb[i], sub))
    if not cands:
        fans[str(i)] = ""
        continue
    fans[str(i)] = ft_generate(val_qs[i], SUBSET_TO_LANG[sub], cands)
    if (n + 1) % 25 == 0:
        json.dump(fans, open(fprog, 'w'))
        elapsed = time.time() - t0
        remaining = elapsed / max(n + 1, 1) * (len(val_sample) - n - 1)
        log(f"  {n+1}/{len(val_sample)} done, ~{remaining/60:.0f}min remaining")

json.dump(fans, open(fprog, 'w'))
log(f"Val generation done: {len(fans)} answers in {(time.time()-t0)/60:.1f}min")

# ===========================================================================
# STEP 3: EVALUATE — Generated vs Retrieval per language (ROUGE)
# ===========================================================================
log(f"\n{'='*60}")
log("STEP 3: Per-language evaluation (ROUGE)")
log(f"{'='*60}")

TEST_MIX = {
    'Eng_Uga': 0.284, 'Aka_Gha': 0.188, 'Eng_Gha': 0.188,
    'Lug_Uga': 0.143, 'Swa_Ken': 0.087, 'Eng_Ken': 0.064,
    'Amh_Eth': 0.023, 'Eng_Eth': 0.023,
}

per_lang = defaultdict(lambda: {'ret_r1': [], 'ret_rl': [], 'gen_r1': [], 'gen_rl': []})

for i in val_sample:
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref:
        continue
    rt = uni_toks(ref)

    # Retrieval top-1
    cands = val_cands_all.get(i)
    if cands:
        ret = cands[0]['answer']
        per_lang[sub]['ret_r1'].append(uni_r1(rt, uni_toks(ret)))
        per_lang[sub]['ret_rl'].append(uni_rl(rt, uni_toks(ret)))

    # Generated
    gen = fans.get(str(i), '')
    if gen:
        per_lang[sub]['gen_r1'].append(uni_r1(rt, uni_toks(gen)))
        per_lang[sub]['gen_rl'].append(uni_rl(rt, uni_toks(gen)))

log(f"\n{'Sub':<12} {'Ret R1':>8} {'Gen R1':>8} {'Δ R1':>7} "
    f"{'Ret RL':>8} {'Gen RL':>8} {'Δ RL':>7} {'Use Gen?':>9}")
log('-' * 80)

gen_better_r1 = {}
gen_better_rl = {}
for sub in sorted(SUBSET_TO_LANG.keys()):
    rr1 = np.mean(per_lang[sub]['ret_r1']) if per_lang[sub]['ret_r1'] else 0
    rrl = np.mean(per_lang[sub]['ret_rl']) if per_lang[sub]['ret_rl'] else 0
    gr1 = np.mean(per_lang[sub]['gen_r1']) if per_lang[sub]['gen_r1'] else 0
    grl = np.mean(per_lang[sub]['gen_rl']) if per_lang[sub]['gen_rl'] else 0

    gen_better_r1[sub] = gr1 > rr1 + 0.005
    gen_better_rl[sub] = grl > rrl + 0.005

    verdict = "GEN" if gen_better_r1[sub] or gen_better_rl[sub] else "RET"
    log(f"  {sub:<12} {rr1:>8.4f} {gr1:>8.4f} {gr1-rr1:>+7.4f} "
        f"{rrl:>8.4f} {grl:>8.4f} {grl-rrl:>+7.4f} {verdict:>9}")

# ===========================================================================
# STEP 4: GENERATE TEST ANSWERS (only needed languages)
# ===========================================================================
log(f"\n{'='*60}")
log("STEP 4: Test generation (2618 queries)")
log(f"{'='*60}")

test_gen_path = OUTPUT_DIR / 'ftqwen_test.json'
test_gen = json.load(open(test_gen_path)) if test_gen_path.exists() else {}
log(f"Already generated: {len(test_gen)}")

t0 = time.time()
for i in tqdm(range(len(test_df)), desc="FT-Qwen test"):
    rid = str(test_df.iloc[i]['ID'])
    if rid in test_gen:
        continue

    q = test_qs[i].strip()
    sub = test_subs[i]
    lang = SUBSET_TO_LANG.get(sub, sub)

    cands = get_same_lang_candidates(q, test_emb[i], sub, k=K_CANDIDATES, exclude_exact=False)
    if not cands:
        test_gen[rid] = "No answer."
        continue

    test_gen[rid] = ft_generate(q, lang, cands)

    if (i + 1) % 100 == 0:
        json.dump(test_gen, open(test_gen_path, 'w'))
        elapsed = time.time() - t0
        done = sum(1 for _ in test_gen if _ not in {})
        log(f"  {i+1}/{len(test_df)}, ~{elapsed/60:.0f}min elapsed")

json.dump(test_gen, open(test_gen_path, 'w'))
log(f"Test generation done: {len(test_gen)} answers in {(time.time()-t0)/60:.1f}min")

# Free GPU
del ft, base
gc.collect(); torch.cuda.empty_cache()
log("GPU freed.")

# ===========================================================================
# STEP 5: BUILD SUBMISSIONS
# ===========================================================================
log(f"\n{'='*60}")
log("STEP 5: Build submissions")
log(f"{'='*60}")

rows_v1 = []  # Safe: MBR for ROUGE, FT-Qwen for LLM
rows_v2 = []  # Compliant: FT-Qwen everywhere
rows_v3 = []  # Compliant: MBR everywhere (fallback)

for i in tqdm(range(len(test_df)), desc="Building submissions"):
    q = test_qs[i].strip()
    sub = test_subs[i]
    rid = str(test_df.iloc[i]['ID'])

    pool_type, alpha, margin = choice.get(sub, ('2leg', 0.05, 99.0))

    # Retrieval candidates
    if pool_type == '4leg':
        cands = union4(q, test_emb[i], gem_test[i], bge_test[i], sub, exclude_exact=False)
    else:
        cands = get_same_lang_candidates(q, test_emb[i], sub, k=K_CANDIDATES, exclude_exact=False)

    if not cands:
        for rows in [rows_v1, rows_v2, rows_v3]:
            rows.append({'ID': test_df.iloc[i]['ID'],
                         'TargetR1F1': 'No answer', 'TargetRLF1': 'No answer',
                         'TargetLLM': 'No answer'})
        continue

    # MBR answers (proven for ROUGE)
    dd, w, u1, uL = uni_prep(cands)
    ans_mbr_r1 = dd[mbr_idx(u1, w, alpha, margin)]
    ans_mbr_rl = dd[mbr_idx(uL, w, alpha, margin)]

    # Stitch for R1 where gated
    sg = uni_stitch_gate.get(sub, {})
    if sg.get('use', False):
        stitch_cands = cands
        if sg.get('pool') == '4leg' and pool_type != '4leg':
            stitch_cands = union4(q, test_emb[i], gem_test[i], bge_test[i], sub, exclude_exact=False)
        ans_stitch_r1 = uni_stitch(stitch_cands, sg['lam'], sub)
    else:
        ans_stitch_r1 = ans_mbr_r1

    # FT-Qwen answer
    gen = test_gen.get(rid, dd[0])

    # V1: SAFE SPLIT — proven ROUGE + FT-Qwen LLM
    rows_v1.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': ans_stitch_r1,
        'TargetRLF1': ans_mbr_rl,
        'TargetLLM': gen,
    })

    # V2: COMPLIANT — FT-Qwen for all
    rows_v2.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': gen, 'TargetRLF1': gen, 'TargetLLM': gen,
    })

    # V3: COMPLIANT — MBR for all (current baseline)
    rows_v3.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': ans_mbr_r1, 'TargetRLF1': ans_mbr_r1, 'TargetLLM': ans_mbr_r1,
    })

# Save
for fname, rows in [
    ('submission_ft_safe_split.csv', rows_v1),
    ('submission_ft_compliant_gen.csv', rows_v2),
    ('submission_ft_compliant_mbr.csv', rows_v3),
]:
    df = pd.DataFrame(rows)[SUB_COLS]
    assert len(df) == len(sample_sub), f"{fname}: {len(df)} vs {len(sample_sub)}"
    df.to_csv(OUTPUT_DIR / fname, index=False)
    log(f"Saved: {fname}")

# ===========================================================================
# SUMMARY
# ===========================================================================
log(f"\n{'='*60}")
log("DONE — SUBMISSION RECOMMENDATIONS")
log(f"{'='*60}")
log(f"""
1. submission_ft_safe_split.csv    ← SUBMIT FIRST
   R1=stitch/MBR (retrieval), RL=MBR (retrieval), LLM=FT-Qwen (generated)
   Why: keeps proven ROUGE, upgrades LLM column

2. submission_ft_compliant_gen.csv ← SUBMIT IF #1 improves over 0.6670
   All columns = FT-Qwen generated (fully open-source + compliant)

3. submission_ft_compliant_mbr.csv ← SAFETY NET
   All columns = MBR retrieval (proven baseline, compliant)

Current LB: 0.6670 | Leader: 0.7285
""")
