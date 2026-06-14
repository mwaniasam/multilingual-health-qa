"""
=============================================================================
DATA-DRIVEN ANALYSIS + PER-LANGUAGE STRATEGY + GEMINI LLM COLUMN
=============================================================================
1. Analyze WHY weak languages score low (Amh 0.04, Aka 0.39, Eng_Gha 0.34)
2. Test per-language retrieval strategies (global vs same-lang vs hybrid)
3. Pick BEST strategy per language based on val data
4. Optionally: use Gemini API for LLM-Judge column
5. Generate optimized submission

Cell 1: from google.colab import drive; drive.mount('/content/drive')
Cell 2: !pip install -q sentence-transformers faiss-cpu rouge-score tqdm
Cell 3: Paste this
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
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime
from collections import defaultdict, Counter

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

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

# ============================================================
# BUILD PER-LANGUAGE INDICES
# ============================================================
log(f"\n{'='*60}")
log("STEP 1: Load model + Build global and per-language indices")
log(f"{'='*60}")

from sentence_transformers import SentenceTransformer

PREFIX = "query: "

if AFRIE5_DIR and AFRIE5_DIR.exists():
    bienc = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0')
    log(f"AfriE5 from Drive: {sum(p.numel() for p in bienc.parameters())/1e6:.0f}M params")
else:
    bienc = SentenceTransformer('McGill-NLP/AfriE5-Large-instruct', device='cuda:0')
    log(f"AfriE5 from HuggingFace")

log("Encoding corpus...")
corpus_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in questions_raw],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

# Global index
global_idx = faiss.IndexFlatIP(corpus_emb.shape[1])
global_idx.add(corpus_emb)
log(f"Global index: {corpus_emb.shape}")

# Per-language indices
lang_indices = {}   # subset -> (faiss_index, global_row_indices)
unique_subsets = sorted(set(subsets_raw))
for sub in unique_subsets:
    mask = [i for i, s in enumerate(subsets_raw) if s == sub]
    if not mask:
        continue
    sub_emb = corpus_emb[mask]
    idx = faiss.IndexFlatIP(sub_emb.shape[1])
    idx.add(sub_emb)
    lang_indices[sub] = (idx, mask)
    log(f"  {sub}: {len(mask)} samples")

# Encode val
log("Encoding val queries...")
val_qs = val_df['input'].fillna('').astype(str).tolist()
val_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in val_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

bienc.cpu(); gc.collect(); torch.cuda.empty_cache()
log("GPU freed.")

# ============================================================
# STEP 2: DEEP ANALYSIS OF EACH LANGUAGE
# ============================================================
log(f"\n{'='*60}")
log("STEP 2: Deep analysis of each language")
log(f"{'='*60}")

# For each val question, get top-50 global candidates and compute oracle
analysis = defaultdict(lambda: {
    'count': 0, 'r1_top1': [], 'rl_top1': [], 'r1_oracle': [], 'rl_oracle': [],
    'oracle_from_same_lang': 0, 'oracle_from_diff_lang': 0,
    'oracle_rank': [], 'avg_q_len': [], 'avg_a_len': [], 'avg_ref_len': [],
    'top1_same_lang': 0, 'top1_diff_lang': 0,
    'examples': [],
})

TOP_K = 50

for i in tqdm(range(len(val_df)), desc="Analyzing val"):
    q = str(val_df.iloc[i]['input']).strip()
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref or not q:
        continue

    D, I = global_idx.search(val_emb[i:i+1], TOP_K + 5)
    candidates = []
    for j in range(TOP_K + 5):
        ci = int(I[0][j])
        if ci >= len(combined): continue
        if str(combined.iloc[ci]['input']).strip() == q: continue
        candidates.append(ci)
        if len(candidates) >= TOP_K: break

    if not candidates:
        continue

    # Top-1
    top1_ci = candidates[0]
    top1_answer = str(combined.iloc[top1_ci]['output'])
    top1_sub = str(combined.iloc[top1_ci]['subset'])
    r_top1 = scorer.score(ref, top1_answer)

    # Oracle (best ROUGE in top-50)
    best_r1, best_rl, best_ci, best_rank = 0, 0, candidates[0], 0
    for rank, ci in enumerate(candidates):
        ca = str(combined.iloc[ci]['output'])
        r = scorer.score(ref, ca)
        combined_r = r['rouge1'].fmeasure + r['rougeL'].fmeasure
        if combined_r > best_r1 + best_rl:
            best_r1, best_rl = r['rouge1'].fmeasure, r['rougeL'].fmeasure
            best_ci, best_rank = ci, rank

    a = analysis[sub]
    a['count'] += 1
    a['r1_top1'].append(r_top1['rouge1'].fmeasure)
    a['rl_top1'].append(r_top1['rougeL'].fmeasure)
    a['r1_oracle'].append(best_r1)
    a['rl_oracle'].append(best_rl)
    a['oracle_rank'].append(best_rank)
    a['avg_q_len'].append(len(q.split()))
    a['avg_a_len'].append(len(top1_answer.split()))
    a['avg_ref_len'].append(len(ref.split()))

    if str(combined.iloc[best_ci]['subset']) == sub:
        a['oracle_from_same_lang'] += 1
    else:
        a['oracle_from_diff_lang'] += 1

    if top1_sub == sub:
        a['top1_same_lang'] += 1
    else:
        a['top1_diff_lang'] += 1

    # Save examples for weak languages
    if len(a['examples']) < 3 and r_top1['rouge1'].fmeasure < 0.4:
        a['examples'].append({
            'question': q[:200],
            'reference': ref[:200],
            'top1_answer': top1_answer[:200],
            'top1_subset': top1_sub,
            'oracle_answer': str(combined.iloc[best_ci]['output'])[:200],
            'oracle_subset': str(combined.iloc[best_ci]['subset']),
            'oracle_rank': best_rank,
            'top1_r1': r_top1['rouge1'].fmeasure,
            'oracle_r1': best_r1,
        })

# Print analysis
log(f"\n{'='*70}")
log(f"{'Subset':<10} {'N':>5} {'Top1 R1':>8} {'Orac R1':>8} {'Gap':>6} "
    f"{'OracRank':>9} {'SameLng%':>9} {'QLen':>5} {'ALen':>5} {'RefLen':>6}")
log(f"{'-'*70}")
for sub in sorted(analysis.keys()):
    a = analysis[sub]
    n = a['count']
    t1_r1 = np.mean(a['r1_top1'])
    or_r1 = np.mean(a['r1_oracle'])
    gap = or_r1 - t1_r1
    avg_rank = np.mean(a['oracle_rank'])
    same_pct = a['oracle_from_same_lang'] / max(n, 1) * 100
    qlen = np.mean(a['avg_q_len'])
    alen = np.mean(a['avg_a_len'])
    rlen = np.mean(a['avg_ref_len'])
    log(f"{sub:<10} {n:>5} {t1_r1:>8.3f} {or_r1:>8.3f} {gap:>+6.3f} "
        f"{avg_rank:>9.1f} {same_pct:>8.1f}% {qlen:>5.0f} {alen:>5.0f} {rlen:>6.0f}")

# Print examples for weak languages
for sub in ['Amh_Eth', 'Aka_Gha', 'Eng_Gha']:
    a = analysis[sub]
    if a['examples']:
        log(f"\n--- Examples for {sub} (why it's weak) ---")
        for j, ex in enumerate(a['examples'][:2]):
            log(f"\n  Example {j+1}:")
            log(f"    Q: {ex['question']}")
            log(f"    Reference:     {ex['reference']}")
            log(f"    Retrieved (R1={ex['top1_r1']:.3f}): {ex['top1_answer']}")
            log(f"      ↑ from subset: {ex['top1_subset']}")
            log(f"    Oracle (R1={ex['oracle_r1']:.3f}, rank={ex['oracle_rank']}): {ex['oracle_answer']}")
            log(f"      ↑ from subset: {ex['oracle_subset']}")

# ============================================================
# STEP 3: TEST PER-LANGUAGE RETRIEVAL STRATEGIES
# ============================================================
log(f"\n{'='*60}")
log("STEP 3: Per-language retrieval strategies")
log(f"{'='*60}")

def retrieve_answer(q_text, q_emb, subset, strategy):
    """Retrieve best answer using given strategy."""
    q_stripped = q_text.strip()

    if strategy == 'global':
        D, I = global_idx.search(q_emb.reshape(1, -1), 10)
        for j in range(10):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            return str(combined.iloc[ci]['output'])

    elif strategy == 'same_lang':
        if subset not in lang_indices:
            return retrieve_answer(q_text, q_emb, subset, 'global')
        idx, mask = lang_indices[subset]
        D, I = idx.search(q_emb.reshape(1, -1), 10)
        for j in range(10):
            li = int(I[0][j])
            if li >= len(mask): continue
            ci = mask[li]
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            return str(combined.iloc[ci]['output'])

    elif strategy == 'same_lang_fallback':
        # Try same language first, fall back to global
        ans = retrieve_answer(q_text, q_emb, subset, 'same_lang')
        if ans:
            return ans
        return retrieve_answer(q_text, q_emb, subset, 'global')

    elif strategy == 'global_top3_longest':
        # Pick the LONGEST answer from top-3 (more content = potentially higher ROUGE recall)
        D, I = global_idx.search(q_emb.reshape(1, -1), 15)
        candidates = []
        for j in range(15):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            candidates.append(str(combined.iloc[ci]['output']))
            if len(candidates) >= 3: break
        if candidates:
            return max(candidates, key=len)

    elif strategy == 'global_top3_best_overlap':
        # Pick answer from top-3 with most word overlap with the QUESTION
        D, I = global_idx.search(q_emb.reshape(1, -1), 15)
        q_words = set(q_stripped.lower().split())
        best_answer, best_overlap = '', 0
        count = 0
        for j in range(15):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            ca = str(combined.iloc[ci]['output'])
            a_words = set(ca.lower().split())
            overlap = len(q_words & a_words)
            if overlap > best_overlap or not best_answer:
                best_overlap = overlap
                best_answer = ca
            count += 1
            if count >= 3: break
        return best_answer

    elif strategy == 'global_concat_top2':
        # Concatenate top-2 answers
        D, I = global_idx.search(q_emb.reshape(1, -1), 15)
        answers = []
        for j in range(15):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q_stripped: continue
            answers.append(str(combined.iloc[ci]['output']))
            if len(answers) >= 2: break
        return ' '.join(answers)

    return "No answer found."

strategies = [
    'global',               # current approach
    'same_lang',            # only same language
    'same_lang_fallback',   # same lang first, then global
    'global_top3_longest',  # longest of top-3
    'global_top3_best_overlap',  # most Q-word overlap from top-3
    'global_concat_top2',   # concatenate top-2
]

# Evaluate each strategy per language
strat_results = {}  # (strategy, subset) -> (r1, rl)

for strat in strategies:
    log(f"\nTesting: {strat}")
    per_lang = defaultdict(lambda: {'r1': [], 'rl': []})

    for i in tqdm(range(len(val_df)), desc=f"  {strat}"):
        q = str(val_df.iloc[i]['input']).strip()
        ref = str(val_df.iloc[i]['output']).strip()
        sub = str(val_df.iloc[i]['subset'])
        if not ref: continue

        answer = retrieve_answer(q, val_emb[i], sub, strat)
        if not answer:
            answer = "No answer found."

        r = scorer.score(ref, answer)
        per_lang[sub]['r1'].append(r['rouge1'].fmeasure)
        per_lang[sub]['rl'].append(r['rougeL'].fmeasure)

    for sub in per_lang:
        r1 = np.mean(per_lang[sub]['r1'])
        rl = np.mean(per_lang[sub]['rl'])
        strat_results[(strat, sub)] = (r1, rl)

# Find best strategy per language
log(f"\n{'='*60}")
log("BEST STRATEGY PER LANGUAGE")
log(f"{'='*60}")

best_per_lang = {}
log(f"\n{'Subset':<12} {'Best Strategy':<26} {'R1':>7} {'RL':>7} {'vs Global R1':>12}")
for sub in sorted(unique_subsets):
    best_strat, best_r1, best_rl = 'global', 0, 0
    for strat in strategies:
        if (strat, sub) in strat_results:
            r1, rl = strat_results[(strat, sub)]
            if r1 + rl > best_r1 + best_rl:
                best_strat, best_r1, best_rl = strat, r1, rl

    global_r1 = strat_results.get(('global', sub), (0, 0))[0]
    diff = best_r1 - global_r1
    best_per_lang[sub] = best_strat
    marker = " ★" if diff > 0.005 else ""
    log(f"  {sub:<12} {best_strat:<26} {best_r1:>7.4f} {best_rl:>7.4f} {diff:>+12.4f}{marker}")

# Calculate overall score with per-language strategy
log(f"\nOverall comparison:")
global_r1s, global_rls = [], []
perlang_r1s, perlang_rls = [], []

for i in range(len(val_df)):
    q = str(val_df.iloc[i]['input']).strip()
    ref = str(val_df.iloc[i]['output']).strip()
    sub = str(val_df.iloc[i]['subset'])
    if not ref: continue

    # Global
    ans_g = retrieve_answer(q, val_emb[i], sub, 'global')
    r_g = scorer.score(ref, ans_g or '')
    global_r1s.append(r_g['rouge1'].fmeasure)
    global_rls.append(r_g['rougeL'].fmeasure)

    # Per-language best
    best_strat = best_per_lang.get(sub, 'global')
    ans_p = retrieve_answer(q, val_emb[i], sub, best_strat)
    r_p = scorer.score(ref, ans_p or '')
    perlang_r1s.append(r_p['rouge1'].fmeasure)
    perlang_rls.append(r_p['rougeL'].fmeasure)

g_r1, g_rl = np.mean(global_r1s), np.mean(global_rls)
p_r1, p_rl = np.mean(perlang_r1s), np.mean(perlang_rls)

log(f"  {'Global (current)':<25} R1={g_r1:.4f} RL={g_rl:.4f}")
log(f"  {'Per-language best':<25} R1={p_r1:.4f} RL={p_rl:.4f}")
log(f"  {'Improvement':<25} R1={p_r1-g_r1:+.4f} RL={p_rl-g_rl:+.4f}")

# ============================================================
# STEP 4: GENERATE SUBMISSIONS
# ============================================================
log(f"\n{'='*60}")
log("STEP 4: Generate submissions")
log(f"{'='*60}")

# Re-load model for test encoding
bienc = SentenceTransformer(str(AFRIE5_DIR) if AFRIE5_DIR and AFRIE5_DIR.exists()
    else 'McGill-NLP/AfriE5-Large-instruct', device='cuda:0')

test_qs = test_df['input'].fillna('').astype(str).tolist()
test_subs = test_df['subset'].fillna('').astype(str).tolist()
log("Encoding test queries...")
test_emb = bienc.encode(
    [f"{PREFIX}{q}" for q in test_qs],
    batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype(np.float32)

bienc.cpu(); gc.collect(); torch.cuda.empty_cache()

# Submission 1: Global (baseline)
log("\nGenerating baseline submission...")
rows_global = []
for i in tqdm(range(len(test_df)), desc="Global"):
    answer = retrieve_answer(test_qs[i], test_emb[i], test_subs[i], 'global')
    rows_global.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
    })
sub_global = pd.DataFrame(rows_global)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
sub_global.to_csv(OUTPUT_DIR / 'submission_global_baseline.csv', index=False)
log("Saved: submission_global_baseline.csv")

# Submission 2: Per-language best strategy
if p_r1 + p_rl > g_r1 + g_rl + 0.001:
    log("\nGenerating per-language submission...")
    rows_perlang = []
    for i in tqdm(range(len(test_df)), desc="Per-lang"):
        sub = test_subs[i]
        strat = best_per_lang.get(sub, 'global')
        answer = retrieve_answer(test_qs[i], test_emb[i], sub, strat)
        rows_perlang.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
        })
    sub_pl = pd.DataFrame(rows_perlang)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_pl.to_csv(OUTPUT_DIR / 'submission_perlang_best.csv', index=False)
    log("Saved: submission_perlang_best.csv")
else:
    log("Per-language strategy didn't improve. Skipping.")

# Submission 3: Per-column optimized
# ROUGE columns get retrieval, LLM column gets the longest/most detailed answer
log("\nGenerating per-column submission (ROUGE=top1, LLM=most detailed)...")
rows_percol = []
for i in tqdm(range(len(test_df)), desc="Per-col"):
    q = test_qs[i]
    sub = test_subs[i]

    # ROUGE columns: best strategy for this language
    strat = best_per_lang.get(sub, 'global')
    ans_rouge = retrieve_answer(q, test_emb[i], sub, strat)

    # LLM column: longest of top-3 (more comprehensive = better LLM judge)
    ans_llm = retrieve_answer(q, test_emb[i], sub, 'global_top3_longest')

    rows_percol.append({
        'ID': test_df.iloc[i]['ID'],
        'TargetR1F1': ans_rouge,
        'TargetRLF1': ans_rouge,
        'TargetLLM': ans_llm or ans_rouge,
    })
sub_pc = pd.DataFrame(rows_percol)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
sub_pc.to_csv(OUTPUT_DIR / 'submission_percol_optimized.csv', index=False)
log("Saved: submission_percol_optimized.csv")

# ============================================================
# STEP 5: GEMINI API FOR LLM COLUMN (if available)
# ============================================================
log(f"\n{'='*60}")
log("STEP 5: Gemini API for LLM column (optional)")
log(f"{'='*60}")

gemini_available = False
try:
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Try Colab secrets
        try:
            from google.colab import userdata
            api_key = userdata.get('GEMINI_API_KEY') or userdata.get('GOOGLE_API_KEY')
        except Exception:
            pass

    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        gemini_available = True
        log("Gemini API available!")
    else:
        log("No Gemini API key found. Skipping Gemini generation.")
        log("To enable: set GEMINI_API_KEY in Colab secrets or environment.")
except Exception as e:
    log(f"Gemini not available: {e}")

if gemini_available:
    import time

    SUBSET_TO_LANG = {
        'Aka_Gha': 'Akan (Ghana)', 'Amh_Eth': 'Amharic (Ethiopia)',
        'Eng_Eth': 'English (Ethiopia)', 'Eng_Gha': 'English (Ghana)',
        'Eng_Ken': 'English (Kenya)', 'Eng_Uga': 'English (Uganda)',
        'Lug_Uga': 'Luganda (Uganda)', 'Swa_Ken': 'Swahili (Kenya)',
    }

    # Progress file for resuming
    progress_file = OUTPUT_DIR / 'gemini_llm_progress.json'
    if progress_file.exists():
        with open(progress_file) as f:
            gemini_answers = json.load(f)
        log(f"Resuming: {len(gemini_answers)} already done")
    else:
        gemini_answers = {}

    for i in tqdm(range(len(test_df)), desc="Gemini LLM"):
        row_id = str(test_df.iloc[i]['ID'])
        if row_id in gemini_answers:
            continue

        q = test_qs[i]
        sub = test_subs[i]
        lang = SUBSET_TO_LANG.get(sub, sub)

        # Get top-3 retrieved answers as context
        D, I = global_idx.search(test_emb[i:i+1], 10)
        contexts = []
        for j in range(10):
            ci = int(I[0][j])
            if ci >= len(combined): continue
            if str(combined.iloc[ci]['input']).strip() == q.strip(): continue
            contexts.append(str(combined.iloc[ci]['output']))
            if len(contexts) >= 3: break

        context_str = "\n".join([f"{k+1}. {c}" for k, c in enumerate(contexts)])

        prompt = f"""You are a multilingual health expert. Answer this health question accurately and comprehensively.

Language: {lang}
Question: {q}

Reference answers for context:
{context_str}

Instructions:
- Answer in {lang} (same language as the question)
- Be thorough, accurate, and culturally appropriate
- Include all relevant medical information
- Use clear, professional language
- Do NOT add disclaimers or meta-commentary"""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=512,
                    )
                )
                gemini_answers[row_id] = response.text.strip()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    gemini_answers[row_id] = contexts[0] if contexts else "No answer."

        # Rate limit (free tier: 15 RPM)
        time.sleep(4.5)

        # Save progress every 50
        if (i + 1) % 50 == 0:
            with open(progress_file, 'w') as f:
                json.dump(gemini_answers, f)
            log(f"  Progress saved: {len(gemini_answers)}/{len(test_df)}")

    # Save final progress
    with open(progress_file, 'w') as f:
        json.dump(gemini_answers, f)

    # Create Gemini hybrid submission
    rows_gemini = []
    for i in range(len(test_df)):
        row_id = str(test_df.iloc[i]['ID'])
        sub = test_subs[i]
        strat = best_per_lang.get(sub, 'global')
        ans_rouge = retrieve_answer(test_qs[i], test_emb[i], sub, strat)

        rows_gemini.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetR1F1': ans_rouge,
            'TargetRLF1': ans_rouge,
            'TargetLLM': gemini_answers.get(row_id, ans_rouge),
        })

    sub_gemini = pd.DataFrame(rows_gemini)[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    sub_gemini.to_csv(OUTPUT_DIR / 'submission_gemini_hybrid.csv', index=False)
    log("Saved: submission_gemini_hybrid.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
log(f"\n{'='*60}")
log("FINAL SUMMARY")
log(f"{'='*60}")

log(f"\nVal scores:")
log(f"  Global baseline:    R1={g_r1:.4f} RL={g_rl:.4f}")
log(f"  Per-language best:  R1={p_r1:.4f} RL={p_rl:.4f} ({p_r1-g_r1:+.4f}/{p_rl-g_rl:+.4f})")

log(f"\nSubmissions:")
for f in sorted(OUTPUT_DIR.glob("submission_*.csv")):
    log(f"  → {f.name}")

log(f"\nStrategy per language: {json.dumps(best_per_lang, indent=2)}")
log(f"\nPrevious best LB: 0.6545")
log(f"\nRecommended submit order:")
log(f"  1. submission_percol_optimized.csv (retrieval for ROUGE, longest for LLM)")
if gemini_available:
    log(f"  2. submission_gemini_hybrid.csv (retrieval for ROUGE, Gemini for LLM)")
log(f"  3. submission_perlang_best.csv (per-language retrieval)")
log("Done!")
