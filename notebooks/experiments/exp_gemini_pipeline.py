"""
=============================================================================
GEMINI EXTRACTIVE COMBINATION + LLM QUALITY PIPELINE
=============================================================================
THE STRATEGY:
- ROUGE columns: Give Gemini top-5 answers → "combine EXACT phrases" → higher recall
- LLM column: Give Gemini top-5 answers → "generate comprehensive answer" → higher quality
- Both: Tested on val FIRST before running on test

WHY this can work:
- Single answer covers ~63% of reference words (R1=0.63)
- Combining 5 answers could cover ~75%+ (if Gemini preserves exact wording)
- LLM-Judge rewards quality/completeness → Gemini excels at this

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm google-genai
Cell 3: Set GEMINI_API_KEY in Colab secrets (left sidebar → 🔑)
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
import traceback
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

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Ghana)', 'Amh_Eth': 'Amharic (Ethiopia)',
    'Eng_Eth': 'English (Ethiopia)', 'Eng_Gha': 'English (Ghana)',
    'Eng_Ken': 'English (Kenya)', 'Eng_Uga': 'English (Uganda)',
    'Lug_Uga': 'Luganda (Uganda)', 'Swa_Ken': 'Swahili (Kenya)',
}

# Per-language best strategy from analysis
BEST_STRATEGY = {
    'Aka_Gha': 'global_top3_best_overlap',
    'Amh_Eth': 'global',
    'Eng_Eth': 'same_lang',
    'Eng_Gha': 'global_top3_best_overlap',
    'Eng_Ken': 'same_lang',
    'Eng_Uga': 'same_lang',
    'Lug_Uga': 'global',
    'Swa_Ken': 'same_lang',
}

log(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# SETUP GEMINI
# ============================================================
log(f"\n{'='*60}")
log("Setting up Gemini API")
log(f"{'='*60}")

api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY') or userdata.get('GOOGLE_API_KEY')
    except Exception:
        pass

if not api_key:
    log("ERROR: No Gemini API key found!")
    log("Set GEMINI_API_KEY in Colab secrets (🔑 icon in left sidebar)")
    raise ValueError("GEMINI_API_KEY not found")

from google import genai
client = genai.Client(api_key=api_key)
log("Gemini API initialized!")

# Test call
try:
    test_resp = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='Say "API working" in exactly 2 words.',
        config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=10),
    )
    log(f"API test: {test_resp.text.strip()}")
except Exception as e:
    log(f"API test failed: {e}")
    raise

# ============================================================
# LOAD AfriE5 + BUILD INDICES
# ============================================================
log(f"\n{'='*60}")
log("Loading AfriE5 + Building indices")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer

PREFIX = "query: "
bienc = SentenceTransformer(str(AFRIE5_DIR) if AFRIE5_DIR and AFRIE5_DIR.exists()
    else 'McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
log(f"AfriE5: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")

log("Encoding corpus...")
corpus_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

global_idx = faiss.IndexFlatIP(corpus_emb.shape[1])
global_idx.add(corpus_emb)

# Per-language indices
lang_indices = {}
for sub in sorted(set(subsets_raw)):
    mask = [i for i, s in enumerate(subsets_raw) if s == sub]
    sub_emb = corpus_emb[mask]
    idx = faiss.IndexFlatIP(sub_emb.shape[1])
    idx.add(sub_emb)
    lang_indices[sub] = (idx, mask)

log("Encoding val + test...")
val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

test_qs = test_df['input'].fillna('').astype(str).tolist()
test_subs = test_df['subset'].fillna('').astype(str).tolist()
test_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in test_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

bienc.cpu(); gc.collect(); torch.cuda.empty_cache()
log("All embeddings done. GPU freed for Gemini API calls.")

# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def get_top_k_answers(q_text, q_emb, subset, k=5):
    """Get top-K answers using per-language best strategy."""
    q_stripped = q_text.strip()
    strategy = BEST_STRATEGY.get(subset, 'global')

    # Always get global candidates
    D_g, I_g = global_idx.search(q_emb.reshape(1, -1), k * 3)
    global_cands = []
    for j in range(k * 3):
        ci = int(I_g[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
        global_cands.append((ci, float(D_g[0][j])))
        if len(global_cands) >= k * 2: break

    # Same-language candidates
    same_lang_cands = []
    if subset in lang_indices:
        idx_l, mask_l = lang_indices[subset]
        D_l, I_l = idx_l.search(q_emb.reshape(1, -1), k * 2)
        for j in range(k * 2):
            li = int(I_l[0][j])
            if li >= len(mask_l): continue
            ci = mask_l[li]
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            same_lang_cands.append((ci, float(D_l[0][j])))
            if len(same_lang_cands) >= k: break

    # Select based on strategy
    if strategy == 'same_lang' and same_lang_cands:
        selected = same_lang_cands[:k]
    elif strategy == 'global_top3_best_overlap':
        q_words = set(q_stripped.lower().split())
        scored = []
        for ci, sim in global_cands[:k*2]:
            ca = str(combined.iloc[ci]['output'])
            a_words = set(ca.lower().split())
            overlap = len(q_words & a_words)
            scored.append((ci, sim, overlap))
        scored.sort(key=lambda x: (-x[2], -x[1]))
        selected = [(ci, sim) for ci, sim, _ in scored[:k]]
    else:
        selected = global_cands[:k]

    results = []
    for ci, sim in selected:
        results.append({
            'question': str(combined.iloc[ci]['input']),
            'answer': str(combined.iloc[ci]['output']),
            'subset': str(combined.iloc[ci]['subset']),
            'similarity': sim,
        })
    return results

# ============================================================
# GEMINI GENERATION FUNCTIONS
# ============================================================
call_count = 0
call_start_time = time.time()

def gemini_call(prompt, temperature=0.3, max_tokens=512, retries=3):
    """Make a Gemini API call with rate limiting and retries."""
    global call_count, call_start_time

    for attempt in range(retries):
        try:
            # Adaptive rate limiting
            call_count += 1
            elapsed = time.time() - call_start_time
            if elapsed < 60 and call_count > 14:
                wait = 60 - elapsed + 1
                time.sleep(wait)
                call_count = 0
                call_start_time = time.time()

            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'quota' in err_str or 'rate' in err_str:
                wait = min(30 * (attempt + 1), 120)
                log(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                call_count = 0
                call_start_time = time.time()
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"  Gemini error after {retries} retries: {e}")
                return None
    return None


def generate_rouge_answer(question, language, contexts):
    """Generate answer optimized for ROUGE (extractive combination)."""
    ctx_str = "\n".join([f"Answer {k+1}: {c['answer']}" for k, c in enumerate(contexts)])

    prompt = f"""You are given a health question and {len(contexts)} reference answers. Create the BEST answer by selecting and combining the most relevant sentences from these references.

CRITICAL RULES:
1. Use ONLY words, phrases, and sentences from the provided reference answers
2. Do NOT paraphrase — copy sentences and phrases EXACTLY as written
3. Do NOT add any new information or your own knowledge
4. Combine the most relevant and complete parts from different answers
5. Keep the answer in the SAME LANGUAGE as the question ({language})
6. Keep the answer similar in length to the individual references
7. If references conflict, use the most detailed/complete version
8. Start directly with the answer — no preamble, no "Based on...", no meta-text

Question: {question}

{ctx_str}

Combined answer (using EXACT text from above references):"""

    return gemini_call(prompt, temperature=0.1, max_tokens=600)


def generate_llm_answer(question, language, contexts):
    """Generate answer optimized for LLM-Judge (quality & completeness)."""
    ctx_str = "\n".join([f"{k+1}. {c['answer']}" for k, c in enumerate(contexts)])

    prompt = f"""You are a professional multilingual health expert. Answer this health question with accuracy, depth, and cultural sensitivity.

Language: {language}
Question: {question}

Reference information:
{ctx_str}

Provide a thorough, well-organized answer that:
- Is in {language} (matching the question's language)
- Covers all important aspects of the health topic
- Uses accurate medical terminology
- Is culturally appropriate
- Is comprehensive but concise
- Directly answers the question without preamble

Answer:"""

    return gemini_call(prompt, temperature=0.3, max_tokens=600)


# ============================================================
# PHASE 1: VAL EVALUATION (200 samples)
# ============================================================
log(f"\n{'='*60}")
log("PHASE 1: Val evaluation (200 samples)")
log(f"{'='*60}")

VAL_SAMPLE = 200
baseline_r1s, baseline_rls = [], []
rouge_gen_r1s, rouge_gen_rls = [], []
per_lang_baseline = defaultdict(lambda: {'r1': [], 'rl': []})
per_lang_gemini = defaultdict(lambda: {'r1': [], 'rl': []})

for i in tqdm(range(VAL_SAMPLE), desc="Val eval"):
    q = val_qs[i]
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    lang = SUBSET_TO_LANG.get(sub, sub)
    if not ref: continue

    contexts = get_top_k_answers(q, val_emb[i], sub, k=5)
    if not contexts: continue

    # Baseline: top-1 answer
    baseline_answer = contexts[0]['answer']
    r_base = scorer.score(ref, baseline_answer)
    baseline_r1s.append(r_base['rouge1'].fmeasure)
    baseline_rls.append(r_base['rougeL'].fmeasure)
    per_lang_baseline[sub]['r1'].append(r_base['rouge1'].fmeasure)
    per_lang_baseline[sub]['rl'].append(r_base['rougeL'].fmeasure)

    # Gemini extractive combination
    gen_answer = generate_rouge_answer(q, lang, contexts)
    if gen_answer:
        r_gen = scorer.score(ref, gen_answer)
        rouge_gen_r1s.append(r_gen['rouge1'].fmeasure)
        rouge_gen_rls.append(r_gen['rougeL'].fmeasure)
        per_lang_gemini[sub]['r1'].append(r_gen['rouge1'].fmeasure)
        per_lang_gemini[sub]['rl'].append(r_gen['rougeL'].fmeasure)
    else:
        rouge_gen_r1s.append(r_base['rouge1'].fmeasure)
        rouge_gen_rls.append(r_base['rougeL'].fmeasure)
        per_lang_gemini[sub]['r1'].append(r_base['rouge1'].fmeasure)
        per_lang_gemini[sub]['rl'].append(r_base['rougeL'].fmeasure)

    if (i + 1) % 50 == 0:
        log(f"  [{i+1}/{VAL_SAMPLE}] Base R1={np.mean(baseline_r1s):.4f} "
            f"Gemini R1={np.mean(rouge_gen_r1s):.4f} "
            f"Δ={np.mean(rouge_gen_r1s)-np.mean(baseline_r1s):+.4f}")

b_r1 = np.mean(baseline_r1s)
b_rl = np.mean(baseline_rls)
g_r1 = np.mean(rouge_gen_r1s)
g_rl = np.mean(rouge_gen_rls)

log(f"\n{'='*60}")
log(f"VAL RESULTS ({VAL_SAMPLE} samples)")
log(f"{'='*60}")
log(f"{'Method':<30} {'R1':>8} {'RL':>8}")
log(f"{'-'*48}")
log(f"{'Retrieval top-1':<30} {b_r1:>8.4f} {b_rl:>8.4f}")
log(f"{'Gemini extractive':<30} {g_r1:>8.4f} {g_rl:>8.4f}")
log(f"{'Improvement':<30} {g_r1-b_r1:>+8.4f} {g_rl-b_rl:>+8.4f}")

log(f"\nPer-language:")
log(f"{'Subset':<12} {'Base R1':>8} {'Gem R1':>8} {'Δ':>7}")
for sub in sorted(set(list(per_lang_baseline.keys()) + list(per_lang_gemini.keys()))):
    br1 = np.mean(per_lang_baseline[sub]['r1']) if per_lang_baseline[sub]['r1'] else 0
    gr1 = np.mean(per_lang_gemini[sub]['r1']) if per_lang_gemini[sub]['r1'] else 0
    log(f"  {sub:<12} {br1:>8.4f} {gr1:>8.4f} {gr1-br1:>+7.4f}")

# Decide: use Gemini for ROUGE or stick with retrieval?
USE_GEMINI_ROUGE = (g_r1 + g_rl) > (b_r1 + b_rl)
log(f"\nDecision: {'USE Gemini for ROUGE columns' if USE_GEMINI_ROUGE else 'KEEP retrieval for ROUGE columns'}")

# Per-language decision: use Gemini only where it helps
USE_GEMINI_PER_LANG = {}
for sub in sorted(set(list(per_lang_baseline.keys()) + list(per_lang_gemini.keys()))):
    br1 = np.mean(per_lang_baseline[sub]['r1']) if per_lang_baseline[sub]['r1'] else 0
    gr1 = np.mean(per_lang_gemini[sub]['r1']) if per_lang_gemini[sub]['r1'] else 0
    USE_GEMINI_PER_LANG[sub] = gr1 > br1 + 0.005
    if USE_GEMINI_PER_LANG[sub]:
        log(f"  ★ {sub}: Use Gemini (+{gr1-br1:.4f})")

# ============================================================
# PHASE 2: GENERATE TEST SUBMISSIONS
# ============================================================
log(f"\n{'='*60}")
log("PHASE 2: Generate test submissions")
log(f"{'='*60}")

# Progress files
rouge_progress_file = OUTPUT_DIR / 'gemini_rouge_progress.json'
llm_progress_file = OUTPUT_DIR / 'gemini_llm_progress.json'

# Load existing progress
rouge_answers = {}
if rouge_progress_file.exists():
    with open(rouge_progress_file) as f:
        rouge_answers = json.load(f)
    log(f"Resuming ROUGE: {len(rouge_answers)} done")

llm_answers = {}
if llm_progress_file.exists():
    with open(llm_progress_file) as f:
        llm_answers = json.load(f)
    log(f"Resuming LLM: {len(llm_answers)} done")

# Generate ROUGE answers (Gemini extractive) — only for languages where it helps
log("\nGenerating ROUGE-optimized answers...")
for i in tqdm(range(len(test_df)), desc="ROUGE gen"):
    row_id = str(test_df.iloc[i]['ID'])
    if row_id in rouge_answers:
        continue

    q = test_qs[i]
    sub = test_subs[i]
    lang = SUBSET_TO_LANG.get(sub, sub)

    contexts = get_top_k_answers(q, test_emb[i], sub, k=5)
    top1_answer = contexts[0]['answer'] if contexts else "No answer."

    # Only use Gemini for languages where it improved val
    if USE_GEMINI_PER_LANG.get(sub, USE_GEMINI_ROUGE):
        gen = generate_rouge_answer(q, lang, contexts)
        rouge_answers[row_id] = gen if gen else top1_answer
    else:
        rouge_answers[row_id] = top1_answer

    if (i + 1) % 100 == 0:
        with open(rouge_progress_file, 'w') as f:
            json.dump(rouge_answers, f)
        log(f"  ROUGE progress saved: {len(rouge_answers)}/{len(test_df)}")

with open(rouge_progress_file, 'w') as f:
    json.dump(rouge_answers, f)
log(f"ROUGE generation complete: {len(rouge_answers)}")

# Generate LLM answers (Gemini quality) — for ALL languages
log("\nGenerating LLM-Judge optimized answers...")
for i in tqdm(range(len(test_df)), desc="LLM gen"):
    row_id = str(test_df.iloc[i]['ID'])
    if row_id in llm_answers:
        continue

    q = test_qs[i]
    sub = test_subs[i]
    lang = SUBSET_TO_LANG.get(sub, sub)

    contexts = get_top_k_answers(q, test_emb[i], sub, k=5)

    gen = generate_llm_answer(q, lang, contexts)
    if gen:
        llm_answers[row_id] = gen
    else:
        llm_answers[row_id] = contexts[0]['answer'] if contexts else "No answer."

    if (i + 1) % 100 == 0:
        with open(llm_progress_file, 'w') as f:
            json.dump(llm_answers, f)
        log(f"  LLM progress saved: {len(llm_answers)}/{len(test_df)}")

with open(llm_progress_file, 'w') as f:
    json.dump(llm_answers, f)
log(f"LLM generation complete: {len(llm_answers)}")

# ============================================================
# PHASE 3: CREATE SUBMISSIONS
# ============================================================
log(f"\n{'='*60}")
log("PHASE 3: Create submissions")
log(f"{'='*60}")

# Submission 1: Full Gemini (ROUGE=extractive, LLM=quality)
rows_full = []
for i in range(len(test_df)):
    row_id = str(test_df.iloc[i]['ID'])
    q = test_qs[i]
    sub = test_subs[i]
    contexts = get_top_k_answers(q, test_emb[i], sub, k=5)
    fallback = contexts[0]['answer'] if contexts else "No answer."

    rows_full.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': rouge_answers.get(row_id, fallback),
        'TargetRLF1': rouge_answers.get(row_id, fallback),
        'TargetLLM': llm_answers.get(row_id, fallback),
    })

sub_full = pd.DataFrame(rows_full)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
assert len(sub_full) == len(sample_sub)
sub_full.to_csv(OUTPUT_DIR / 'submission_gemini_full.csv', index=False)
log("Saved: submission_gemini_full.csv (ROUGE=extractive, LLM=quality)")

# Submission 2: Hybrid (ROUGE=retrieval, LLM=Gemini quality)
rows_hybrid = []
for i in range(len(test_df)):
    row_id = str(test_df.iloc[i]['ID'])
    q = test_qs[i]
    sub = test_subs[i]
    contexts = get_top_k_answers(q, test_emb[i], sub, k=5)
    retrieval_ans = contexts[0]['answer'] if contexts else "No answer."

    rows_hybrid.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': retrieval_ans,
        'TargetRLF1': retrieval_ans,
        'TargetLLM': llm_answers.get(row_id, retrieval_ans),
    })

sub_hybrid = pd.DataFrame(rows_hybrid)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
sub_hybrid.to_csv(OUTPUT_DIR / 'submission_gemini_hybrid_v2.csv', index=False)
log("Saved: submission_gemini_hybrid_v2.csv (ROUGE=retrieval, LLM=Gemini)")

# Submission 3: Per-language smart (Gemini ROUGE where it helps, retrieval where it doesn't)
rows_smart = []
for i in range(len(test_df)):
    row_id = str(test_df.iloc[i]['ID'])
    q = test_qs[i]
    sub = test_subs[i]
    contexts = get_top_k_answers(q, test_emb[i], sub, k=5)
    retrieval_ans = contexts[0]['answer'] if contexts else "No answer."

    use_gemini_rouge = USE_GEMINI_PER_LANG.get(sub, False)
    rouge_ans = rouge_answers.get(row_id, retrieval_ans) if use_gemini_rouge else retrieval_ans

    rows_smart.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': rouge_ans,
        'TargetRLF1': rouge_ans,
        'TargetLLM': llm_answers.get(row_id, retrieval_ans),
    })

sub_smart = pd.DataFrame(rows_smart)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
sub_smart.to_csv(OUTPUT_DIR / 'submission_gemini_smart.csv', index=False)
log("Saved: submission_gemini_smart.csv (ROUGE=per-lang smart, LLM=Gemini)")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*60}")
log("DONE — ALL SUBMISSIONS READY")
log(f"{'='*60}")
log(f"\nVal ROUGE results ({VAL_SAMPLE} samples):")
log(f"  Retrieval:  R1={b_r1:.4f} RL={b_rl:.4f}")
log(f"  Gemini ext: R1={g_r1:.4f} RL={g_rl:.4f} ({g_r1-b_r1:+.4f}/{g_rl-b_rl:+.4f})")
log(f"\nSubmissions saved:")
log(f"  1. submission_gemini_full.csv     — Gemini for everything")
log(f"  2. submission_gemini_hybrid_v2.csv — retrieval ROUGE + Gemini LLM")
log(f"  3. submission_gemini_smart.csv    — per-lang smart + Gemini LLM")
log(f"  4. submission_perlang_best.csv    — per-lang retrieval (from earlier)")
log(f"\nPrevious best LB: 0.6545")
log(f"Languages using Gemini for ROUGE: {[s for s,v in USE_GEMINI_PER_LANG.items() if v]}")
log(f"\nRecommended submit order:")
log(f"  1. submission_gemini_smart.csv (safest — only uses Gemini where it helps)")
log(f"  2. submission_gemini_hybrid_v2.csv (retrieval ROUGE + Gemini LLM)")
log(f"  3. submission_gemini_full.csv (most aggressive)")
