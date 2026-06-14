"""
=============================================================================
MBR SELECTION + THREE-COLUMN OPTIMIZATION
=============================================================================
THE KEY INSIGHT (from Claude):
  Don't predict ROUGE against an unknown reference.
  Pick the candidate that AGREES MOST with the other candidates.
  Valid answers to similar questions overlap with each other.

Strategy:
  TargetR1F1 → MBR with ROUGE-1 utility (unigram consensus)
  TargetRLF1 → MBR with ROUGE-L utility (sequential overlap consensus)
  TargetLLM  → Gemini-generated OR retrieval (whichever wins on val)

Same-language filtering + guarded override + per-language margin tuning.

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: import os; os.environ['GEMINI_API_KEY'] = 'YOUR_KEY_HERE'
Cell 3: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm google-genai
Cell 4: Paste this
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
import json
import time
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# PATHS + DATA
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

log("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df   = pd.read_csv(DATA_DIR / 'Val.csv')
test_df  = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])
combined = combined.reset_index(drop=True)
questions_raw = combined['input'].fillna('').astype(str).tolist()
answers_raw   = combined['output'].fillna('').astype(str).tolist()
subsets_raw   = combined['subset'].fillna('').astype(str).tolist()

log(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Combined: {len(combined)}")

if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Ghana)', 'Amh_Eth': 'Amharic (Ethiopia)',
    'Eng_Eth': 'English (Ethiopia)', 'Eng_Gha': 'English (Ghana)',
    'Eng_Ken': 'English (Kenya)', 'Eng_Uga': 'English (Uganda)',
    'Lug_Uga': 'Luganda (Uganda)', 'Swa_Ken': 'Swahili (Kenya)',
}

# ============================================================
# STEP 0: AMHARIC TOKENIZATION CHECK
# ============================================================
log(f"\n{'='*60}")
log("STEP 0: Amharic ROUGE tokenization check")
log(f"{'='*60}")

scorer_r1 = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
scorer_rl = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
scorer_both = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

# Test Amharic tokenization
amh_test = "የጤና ጥያቄ ምን ይመስላል"
r = scorer_r1.score(amh_test, amh_test)
log(f"Amharic self-ROUGE: {r['rouge1'].fmeasure:.4f}")
if r['rouge1'].fmeasure < 0.5:
    log("CONFIRMED: ROUGE scorer cannot tokenize Ge'ez script!")
    log("Amharic R1≈0.04 is from Latin tokens (drug names, numbers) only.")
    log("Strategy: for Amharic, MBR will automatically pick candidates sharing Latin tokens.")
    log("Focus LLM-Judge column for Amharic gains (judge reads Ge'ez fine).")
else:
    log("Amharic tokenization works — ROUGE can process Ge'ez script.")

# ============================================================
# STEP 1: LOAD AfriE5 + BUILD INDICES
# ============================================================
log(f"\n{'='*60}")
log("STEP 1: Load model + Build per-language indices")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer
PREFIX = "query: "

bienc = SentenceTransformer(str(AFRIE5_DIR) if AFRIE5_DIR and AFRIE5_DIR.exists()
    else 'McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
log(f"AfriE5: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

log("Encoding corpus questions...")
corpus_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

# Global index
global_idx = faiss.IndexFlatIP(corpus_emb.shape[1])
global_idx.add(corpus_emb)

# Per-language indices (CRITICAL: same-language filtering)
lang_indices = {}
for sub in sorted(set(subsets_raw)):
    mask = [i for i, s in enumerate(subsets_raw) if s == sub]
    sub_emb = corpus_emb[mask]
    idx = faiss.IndexFlatIP(sub_emb.shape[1])
    idx.add(sub_emb)
    lang_indices[sub] = (idx, mask)
    log(f"  {sub}: {len(mask)} samples")

log("Encoding val queries...")
val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

log("Encoding test queries...")
test_qs = test_df['input'].fillna('').astype(str).tolist()
test_subs = test_df['subset'].fillna('').astype(str).tolist()
test_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in test_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

bienc.cpu(); gc.collect(); torch.cuda.empty_cache()
log("GPU freed.")

# ============================================================
# STEP 2: MBR SELECTION FUNCTION
# ============================================================
def get_same_lang_candidates(q_text, q_emb, subset, k=20):
    """Get top-k candidates from SAME LANGUAGE only."""
    q_stripped = q_text.strip()

    if subset in lang_indices:
        idx, mask = lang_indices[subset]
        D, I = idx.search(q_emb.reshape(1, -1), k + 5)
        results = []
        for j in range(k + 5):
            li = int(I[0][j])
            if li >= len(mask): continue
            ci = mask[li]
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            results.append({
                'answer': str(combined.iloc[ci]['output']),
                'sim': float(D[0][j]),
                'idx': ci,
            })
            if len(results) >= k: break
        return results

    # Fallback: global index
    D, I = global_idx.search(q_emb.reshape(1, -1), k + 5)
    results = []
    for j in range(k + 5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
        results.append({
            'answer': str(combined.iloc[ci]['output']),
            'sim': float(D[0][j]),
            'idx': ci,
        })
        if len(results) >= k: break
    return results


def mbr_select(cands, ret_scores, metric='rouge1', alpha=0.15, margin=0.02):
    """MBR: pick the candidate that agrees most with OTHER candidates.
    
    cands:      list of answer strings (top-1 first)
    ret_scores: retrieval similarity scores
    metric:     'rouge1' or 'rougeL'
    alpha:      weight for retrieval score tie-breaking
    margin:     minimum utility advantage to override top-1
    """
    if len(cands) <= 1:
        return cands[0] if cands else ""

    # Sharpen weights toward retrieval confidence
    w = np.exp(np.array(ret_scores) * 5)
    w /= w.sum()

    # Deduplicate but track counts
    seen = {}
    dedup_idx = []
    weights = []
    for i, c in enumerate(cands):
        c_norm = c.strip().lower()
        if c_norm in seen:
            weights[seen[c_norm]] += w[i]  # add weight to existing
        else:
            seen[c_norm] = len(dedup_idx)
            dedup_idx.append(i)
            weights.append(w[i])

    # Use deduped candidates
    dd_cands = [cands[i] for i in dedup_idx]
    dd_weights = np.array(weights)
    dd_weights /= dd_weights.sum()

    if len(dd_cands) == 1:
        return dd_cands[0]

    scorer_m = scorer_r1 if metric == 'rouge1' else scorer_rl

    util = np.zeros(len(dd_cands))
    for i, ci in enumerate(dd_cands):
        for j, cj in enumerate(dd_cands):
            if i != j:
                r = scorer_m.score(cj, ci)
                util[i] += dd_weights[j] * r[metric].fmeasure

    # Tie-break toward retrieval rank
    util += alpha * dd_weights

    best = int(np.argmax(util))
    top1_idx = 0  # first candidate is always top-1

    # Guarded override: keep top-1 unless consensus clearly prefers another
    if best == top1_idx:
        return dd_cands[top1_idx]
    elif util[best] - util[top1_idx] > margin:
        return dd_cands[best]
    else:
        return dd_cands[top1_idx]

# ============================================================
# STEP 3: TUNE MBR MARGINS ON VAL (per language)
# ============================================================
log(f"\n{'='*60}")
log("STEP 3: MBR margin tuning on val (per language)")
log(f"{'='*60}")

K_CANDIDATES = 15  # 10-20 as recommended

# First, evaluate baseline + MBR with different margins
margins_to_test = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
alphas_to_test = [0.05, 0.10, 0.15, 0.20]

# Pre-compute all val candidates (same-language)
log("Pre-computing val candidates (same-language)...")
val_cands_all = []
for i in tqdm(range(len(val_df)), desc="Val candidates"):
    q = val_qs[i].strip()
    sub = str(val_df.iloc[i]['subset'])
    cands = get_same_lang_candidates(q, val_emb[i], sub, k=K_CANDIDATES)
    val_cands_all.append(cands)

# Grid search over alpha and margin, per language
log("\nGrid search: alpha × margin per language...")
best_params = {}  # subset -> (best_alpha, best_margin, best_score)

for sub in sorted(SUBSET_TO_LANG.keys()):
    sub_indices = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset']) == sub]
    if not sub_indices:
        continue

    best_combo_score = -1
    best_alpha, best_margin = 0.15, 0.02

    for alpha in alphas_to_test:
        for margin in margins_to_test:
            r1_scores, rl_scores = [], []

            for i in sub_indices:
                ref = str(val_df.iloc[i]['output']).strip()
                if not ref: continue
                cands = val_cands_all[i]
                if not cands: continue

                answers = [c['answer'] for c in cands]
                sims = [c['sim'] for c in cands]

                mbr_r1 = mbr_select(answers, sims, 'rouge1', alpha, margin)
                mbr_rl = mbr_select(answers, sims, 'rougeL', alpha, margin)

                r1 = scorer_both.score(ref, mbr_r1)
                rl = scorer_both.score(ref, mbr_rl)
                r1_scores.append(r1['rouge1'].fmeasure)
                rl_scores.append(rl['rougeL'].fmeasure)

            avg_r1 = np.mean(r1_scores) if r1_scores else 0
            avg_rl = np.mean(rl_scores) if rl_scores else 0
            combo = avg_r1 + avg_rl

            if combo > best_combo_score:
                best_combo_score = combo
                best_alpha = alpha
                best_margin = margin

    best_params[sub] = (best_alpha, best_margin)

# Evaluate with best params
log(f"\n{'='*60}")
log("RESULTS: MBR vs Baseline (per language, best params)")
log(f"{'='*60}")

baseline_total_r1, baseline_total_rl = [], []
mbr_total_r1, mbr_total_rl = [], []
mbr_r1_col, mbr_rl_col = [], []  # separate columns

log(f"\n{'Sub':<10} {'α':>5} {'τ':>6} {'Base R1':>8} {'MBR R1':>8} {'Δ R1':>7} "
    f"{'Base RL':>8} {'MBR RL':>8} {'Δ RL':>7}")
log('-' * 78)

for sub in sorted(SUBSET_TO_LANG.keys()):
    sub_indices = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset']) == sub]
    if not sub_indices: continue

    alpha, margin = best_params[sub]
    base_r1s, base_rls = [], []
    mbr_r1s, mbr_rls = [], []
    mbr_r1_only, mbr_rl_only = [], []

    for i in sub_indices:
        ref = str(val_df.iloc[i]['output']).strip()
        if not ref: continue
        cands = val_cands_all[i]
        if not cands: continue

        answers = [c['answer'] for c in cands]
        sims = [c['sim'] for c in cands]

        # Baseline: top-1
        r_base = scorer_both.score(ref, answers[0])
        base_r1s.append(r_base['rouge1'].fmeasure)
        base_rls.append(r_base['rougeL'].fmeasure)
        baseline_total_r1.append(r_base['rouge1'].fmeasure)
        baseline_total_rl.append(r_base['rougeL'].fmeasure)

        # MBR: separate for R1 and RL columns
        ans_r1 = mbr_select(answers, sims, 'rouge1', alpha, margin)
        ans_rl = mbr_select(answers, sims, 'rougeL', alpha, margin)

        r_mbr_r1 = scorer_both.score(ref, ans_r1)
        r_mbr_rl = scorer_both.score(ref, ans_rl)

        # R1 column scored on rouge1
        mbr_r1_only.append(r_mbr_r1['rouge1'].fmeasure)
        # RL column scored on rougeL
        mbr_rl_only.append(r_mbr_rl['rougeL'].fmeasure)

        # For comparison: MBR R1 answer scored on both
        mbr_r1s.append(r_mbr_r1['rouge1'].fmeasure)
        mbr_rls.append(r_mbr_rl['rougeL'].fmeasure)
        mbr_total_r1.append(r_mbr_r1['rouge1'].fmeasure)
        mbr_total_rl.append(r_mbr_rl['rougeL'].fmeasure)

    br1, brl = np.mean(base_r1s), np.mean(base_rls)
    mr1, mrl = np.mean(mbr_r1s), np.mean(mbr_rls)
    dr1, drl = mr1 - br1, mrl - brl
    marker = " ★" if dr1 > 0.005 or drl > 0.005 else ""
    log(f"  {sub:<10} {alpha:>5.2f} {margin:>6.3f} {br1:>8.4f} {mr1:>8.4f} {dr1:>+7.4f} "
        f"{brl:>8.4f} {mrl:>8.4f} {drl:>+7.4f}{marker}")

overall_base_r1 = np.mean(baseline_total_r1)
overall_base_rl = np.mean(baseline_total_rl)
overall_mbr_r1 = np.mean(mbr_total_r1)
overall_mbr_rl = np.mean(mbr_total_rl)

log(f"\n  {'OVERALL':<10} {'':>5} {'':>6} {overall_base_r1:>8.4f} {overall_mbr_r1:>8.4f} "
    f"{overall_mbr_r1-overall_base_r1:>+7.4f} {overall_base_rl:>8.4f} {overall_mbr_rl:>8.4f} "
    f"{overall_mbr_rl-overall_base_rl:>+7.4f}")

# Estimated LB improvement
est_r1 = overall_mbr_r1
est_rl = overall_mbr_rl
est_llm = 0.775  # unchanged for now
est_lb = (est_r1 + est_rl + est_llm) / 3
log(f"\n  Estimated LB (MBR ROUGE + baseline LLM): {est_lb:.4f}")
log(f"  Previous best LB: 0.6545")

# ============================================================
# STEP 4: GENERATE TEST SUBMISSIONS
# ============================================================
log(f"\n{'='*60}")
log("STEP 4: Generate test submissions")
log(f"{'='*60}")

# Reload model for test encoding (already done above)
log("Generating MBR submission for test...")
rows_mbr = []
for i in tqdm(range(len(test_df)), desc="MBR test"):
    q = test_qs[i].strip()
    sub = test_subs[i]

    cands = get_same_lang_candidates(q, test_emb[i], sub, k=K_CANDIDATES)
    if not cands:
        rows_mbr.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': 'No answer', 'TargetRLF1': 'No answer', 'TargetLLM': 'No answer'
        })
        continue

    answers = [c['answer'] for c in cands]
    sims = [c['sim'] for c in cands]
    alpha, margin = best_params.get(sub, (0.15, 0.02))

    # THREE DIFFERENT ANSWERS for three columns
    ans_r1 = mbr_select(answers, sims, 'rouge1', alpha, margin)
    ans_rl = mbr_select(answers, sims, 'rougeL', alpha, margin)
    ans_llm = answers[0]  # top-1 for LLM (will be replaced by Gemini later)

    rows_mbr.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': ans_r1,
        'TargetRLF1': ans_rl,
        'TargetLLM': ans_llm,
    })

sub_mbr = pd.DataFrame(rows_mbr)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_mbr) == len(sample_sub), f"Length mismatch: {len(sub_mbr)} vs {len(sample_sub)}"
sub_mbr.to_csv(OUTPUT_DIR / 'submission_mbr.csv', index=False)
log("Saved: submission_mbr.csv (MBR ROUGE, retrieval LLM)")

# Also save baseline with same-language filtering for comparison
rows_samelang = []
for i in range(len(test_df)):
    q = test_qs[i].strip()
    sub = test_subs[i]
    cands = get_same_lang_candidates(q, test_emb[i], sub, k=5)
    ans = cands[0]['answer'] if cands else 'No answer'
    rows_samelang.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': ans, 'TargetRLF1': ans, 'TargetLLM': ans,
    })
sub_sl = pd.DataFrame(rows_samelang)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
sub_sl.to_csv(OUTPUT_DIR / 'submission_samelang_top1.csv', index=False)
log("Saved: submission_samelang_top1.csv (same-lang top-1 baseline)")

# ============================================================
# STEP 5: GEMINI LLM COLUMN (optional, runs overnight)
# ============================================================
log(f"\n{'='*60}")
log("STEP 5: Gemini LLM column")
log(f"{'='*60}")

api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY') or userdata.get('GOOGLE_API_KEY')
    except Exception:
        pass

GEMINI_OK = False
if api_key:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        test_r = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Say OK',
            config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=5),
        )
        log(f"Gemini API test: {test_r.text.strip()}")
        GEMINI_OK = True
    except Exception as e:
        log(f"Gemini API failed: {e}")
else:
    log("No Gemini API key. Skipping Gemini LLM column.")
    log("Set GEMINI_API_KEY in env or Colab secrets to enable.")

if GEMINI_OK:
    call_count, call_start = 0, time.time()

    def gemini_call(prompt, temp=0.3, max_tok=600, retries=3):
        global call_count, call_start
        for attempt in range(retries):
            try:
                call_count += 1
                elapsed = time.time() - call_start
                if elapsed < 60 and call_count > 14:
                    time.sleep(61 - elapsed)
                    call_count, call_start = 0, time.time()
                resp = client.models.generate_content(
                    model='gemini-2.0-flash', contents=prompt,
                    config=genai.types.GenerateContentConfig(temperature=temp, max_output_tokens=max_tok),
                )
                return resp.text.strip()
            except Exception as e:
                if '429' in str(e).lower() or 'quota' in str(e).lower():
                    wait = min(30 * (attempt + 1), 120)
                    log(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    call_count, call_start = 0, time.time()
                elif attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        return None

    # Progress file (resumable)
    llm_prog = OUTPUT_DIR / 'gemini_mbr_llm_prog.json'
    llm_ans = json.load(open(llm_prog)) if llm_prog.exists() else {}
    log(f"Resume: {len(llm_ans)} LLM answers done")

    log("\nGenerating Gemini LLM answers for test...")
    for i in tqdm(range(len(test_df)), desc="Gemini LLM"):
        rid = str(test_df.iloc[i]['ID'])
        if rid in llm_ans:
            continue

        q = test_qs[i]
        sub = test_subs[i]
        lang = SUBSET_TO_LANG.get(sub, sub)

        cands = get_same_lang_candidates(q, test_emb[i], sub, k=5)
        if not cands:
            llm_ans[rid] = "No answer."
            continue

        ctx_str = "\n".join([f"{k+1}. {c['answer']}" for k, c in enumerate(cands)])

        prompt = f"""You are a professional multilingual health expert. Answer this health question in {lang}.

Question: {q}

Reference answers for context:
{ctx_str}

Base your answer on the references above. Be complete, medically accurate, culturally appropriate, and direct.
Do not mention the references. Answer in {lang}:"""

        gen = gemini_call(prompt, temp=0.3)
        llm_ans[rid] = gen if gen else cands[0]['answer']

        if (i+1) % 100 == 0:
            json.dump(llm_ans, open(llm_prog, 'w'))
            log(f"  Progress saved: {len(llm_ans)}/{len(test_df)}")

    json.dump(llm_ans, open(llm_prog, 'w'))
    log(f"Gemini LLM complete: {len(llm_ans)} answers")

    # Create MBR + Gemini submission
    rows_mbr_gem = []
    for i in range(len(test_df)):
        rid = str(test_df.iloc[i]['ID'])
        mbr_row = rows_mbr[i]
        rows_mbr_gem.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': mbr_row['TargetR1F1'],
            'TargetRLF1': mbr_row['TargetRLF1'],
            'TargetLLM': llm_ans.get(rid, mbr_row['TargetLLM']),
        })

    sub_mbr_gem = pd.DataFrame(rows_mbr_gem)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_mbr_gem.to_csv(OUTPUT_DIR / 'submission_mbr_gemini.csv', index=False)
    log("Saved: submission_mbr_gemini.csv (MBR ROUGE + Gemini LLM)")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*60}")
log("DONE — MBR SUBMISSIONS READY")
log(f"{'='*60}")

log(f"\nVal scores:")
log(f"  Baseline (global top-1):  R1={overall_base_r1:.4f} RL={overall_base_rl:.4f}")
log(f"  MBR (same-lang, tuned):   R1={overall_mbr_r1:.4f} RL={overall_mbr_rl:.4f}")
log(f"  Improvement:              R1={overall_mbr_r1-overall_base_r1:+.4f} RL={overall_mbr_rl-overall_base_rl:+.4f}")

log(f"\nBest params per language:")
for sub in sorted(best_params):
    alpha, margin = best_params[sub]
    log(f"  {sub}: α={alpha:.2f}, τ={margin:.3f}")

log(f"\nSubmissions:")
log(f"  1. submission_mbr.csv              (MBR R1 + MBR RL + retrieval LLM)")
if GEMINI_OK:
    log(f"  2. submission_mbr_gemini.csv       (MBR R1 + MBR RL + Gemini LLM)")
log(f"  3. submission_samelang_top1.csv     (same-lang top-1, for comparison)")

log(f"\nPrevious best LB: 0.6545")
log(f"Estimated LB (MBR + baseline LLM): {est_lb:.4f}")
if GEMINI_OK:
    est_lb_gem = (overall_mbr_r1 + overall_mbr_rl + 0.85) / 3
    log(f"Estimated LB (MBR + Gemini LLM):    {est_lb_gem:.4f}")

log(f"\nSubmit order:")
log(f"  1. submission_mbr.csv (isolates MBR gain on ROUGE — one variable changed)")
if GEMINI_OK:
    log(f"  2. submission_mbr_gemini.csv (adds Gemini LLM)")
log(f"\nDone!")
