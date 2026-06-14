"""
=============================================================================
HYBRID SUBMISSION: Gemini-enhanced per-metric optimization
=============================================================================
NO GPU NEEDED — runs on CPU with Gemini API only.

Current metric breakdown (0.6410 overall):
  ROUGE-1 F1: 0.6271  (decent)
  ROUGE-L F1: 0.5607  (BOTTLENECK — too much extra text hurts precision)
  LLM Judge:  0.7749  (strong — try to push higher)

Strategy:
  TargetR1F1 → keep original retrieved answer (already 0.6271)
  TargetRLF1 → Gemini CONDENSES answer (boost precision → improve ROUGE-L)
  TargetLLM  → Gemini POLISHES answer (push 0.77 → 0.85)
=============================================================================
"""
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from google import genai

# ============================================================
# CONFIG
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.5-flash"

DATA_DIR = Path('/home/mwaniasamuel/multilingual-health-qa/data/raw/')
BEST_CSV = Path('/home/mwaniasamuel/multilingual-health-qa/submissions/exp_iterative_HN_round2.csv')
OUTPUT_DIR = Path('/home/mwaniasamuel/multilingual-health-qa/submissions/')
PROGRESS_FILE = OUTPUT_DIR / 'gemini_hybrid_progress.json'

SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Twi)',
    'Amh_Eth': 'Amharic',
    'Eng_Eth': 'English',
    'Eng_Gha': 'English',
    'Eng_Ken': 'English',
    'Eng_Uga': 'English',
    'Lug_Uga': 'Luganda',
    'Swa_Ken': 'Swahili',
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def call_gemini(prompt, max_retries=5):
    """Call Gemini with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=600,
                ),
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                if '503' in err_str or 'UNAVAILABLE' in err_str:
                    wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s for 503
                else:
                    wait = 2 ** attempt
                log(f"  Retry {attempt+1}/{max_retries}: {err_str[:80]} (wait {wait}s)")
                time.sleep(wait)
            else:
                log(f"  FAILED after {max_retries} attempts: {err_str[:100]}")
                return None
    return None


def condense_for_rouge_l(question, answer, language):
    """Extract the most relevant sentences for ROUGE-L (improve precision)."""
    prompt = f"""You are given a health question and a long answer. Your task is to EXTRACT only the most essential 2-3 sentences that directly answer the question.

CRITICAL RULES:
- Keep the EXACT original wording — do NOT paraphrase or rewrite
- Only remove sentences that are tangential, repetitive, or off-topic
- The response must be in {language}
- If the answer is already short (under 200 characters), return it unchanged
- Do NOT add any introduction, greeting, or explanation

Question: {question}

Full answer: {answer}

Extracted key sentences:"""
    result = call_gemini(prompt)
    if result and len(result) > 10:
        return result
    return answer  # fallback to original


def polish_for_llm_judge(question, answer, language):
    """Generate a polished, high-quality answer for LLM-as-judge."""
    prompt = f"""You are a trusted health expert. Answer this health question using the reference information below. Your answer should be:

1. Factually accurate — use ONLY information from the reference
2. Written in {language} — match the language of the question exactly
3. Well-structured and clear
4. Complete but concise (150-300 words)
5. Culturally sensitive and appropriate

Question: {question}

Reference information: {answer}

Your expert answer in {language}:"""
    result = call_gemini(prompt)
    if result and len(result) > 20:
        return result
    return answer  # fallback to original


# ============================================================
# LOAD DATA
# ============================================================
log("Loading data...")
test_df = pd.read_csv(DATA_DIR / 'Test.csv')
best_sub = pd.read_csv(BEST_CSV)
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

assert len(test_df) == len(best_sub), f"Mismatch: test={len(test_df)}, csv={len(best_sub)}"
log(f"Test questions: {len(test_df)}")
log(f"Best submission loaded: {BEST_CSV.name}")

# Load progress if exists
progress = {}
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    log(f"Resuming from {len(progress)} completed questions")

# ============================================================
# PROCESS EACH QUESTION
# ============================================================
log("\nProcessing questions with Gemini...")
condensed_answers = {}  # for ROUGE-L
polished_answers = {}   # for LLM judge

for idx in range(len(test_df)):
    row_id = str(test_df.iloc[idx]['ID'])
    question = str(test_df.iloc[idx]['input']).strip()
    subset = test_df.iloc[idx]['subset']
    language = SUBSET_TO_LANG.get(subset, 'English')
    original_answer = str(best_sub.iloc[idx]['TargetR1F1']).strip()

    # Skip if already processed
    if row_id in progress:
        condensed_answers[row_id] = progress[row_id].get('condensed', original_answer)
        polished_answers[row_id] = progress[row_id].get('polished', original_answer)
        continue

    if idx % 50 == 0:
        log(f"  [{idx}/{len(test_df)}] Processing {subset} ({language})...")

    # 1. Condense for ROUGE-L
    condensed = condense_for_rouge_l(question, original_answer, language)
    condensed_answers[row_id] = condensed

    # 2. Polish for LLM judge
    polished = polish_for_llm_judge(question, original_answer, language)
    polished_answers[row_id] = polished

    # Save progress
    progress[row_id] = {
        'condensed': condensed,
        'polished': polished,
    }

    # Save progress every 100 questions
    if (idx + 1) % 100 == 0:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, ensure_ascii=False)
        log(f"  Progress saved: {idx+1}/{len(test_df)}")

    # Small delay to be safe with rate limits
    time.sleep(0.1)

# Final save
with open(PROGRESS_FILE, 'w') as f:
    json.dump(progress, f, ensure_ascii=False)
log(f"✅ All {len(test_df)} questions processed!")

# ============================================================
# CREATE SUBMISSION VARIANTS
# ============================================================
log("\nCreating submission variants...")

# Variant A: Condensed ROUGE-L only (safest change)
sub_a = best_sub.copy()
for idx in range(len(test_df)):
    row_id = str(test_df.iloc[idx]['ID'])
    sub_a.iloc[idx, sub_a.columns.get_loc('TargetRLF1')] = condensed_answers[row_id]
fname_a = 'hybrid_condensed_rl.csv'
sub_a.to_csv(OUTPUT_DIR / fname_a, index=False)
log(f"✅ Variant A: {fname_a} (condensed ROUGE-L only)")

# Variant B: Polished LLM only (safest change)
sub_b = best_sub.copy()
for idx in range(len(test_df)):
    row_id = str(test_df.iloc[idx]['ID'])
    sub_b.iloc[idx, sub_b.columns.get_loc('TargetLLM')] = polished_answers[row_id]
fname_b = 'hybrid_polished_llm.csv'
sub_b.to_csv(OUTPUT_DIR / fname_b, index=False)
log(f"✅ Variant B: {fname_b} (polished LLM only)")

# Variant C: BOTH condensed ROUGE-L + polished LLM (full hybrid)
sub_c = best_sub.copy()
for idx in range(len(test_df)):
    row_id = str(test_df.iloc[idx]['ID'])
    sub_c.iloc[idx, sub_c.columns.get_loc('TargetRLF1')] = condensed_answers[row_id]
    sub_c.iloc[idx, sub_c.columns.get_loc('TargetLLM')] = polished_answers[row_id]
fname_c = 'hybrid_full.csv'
sub_c.to_csv(OUTPUT_DIR / fname_c, index=False)
log(f"✅ Variant C: {fname_c} (condensed RL + polished LLM)")

# Quick stats
orig_len = best_sub['TargetRLF1'].str.len().mean()
cond_len = sub_a['TargetRLF1'].str.len().mean()
log(f"\nAnswer length: original={orig_len:.0f} → condensed={cond_len:.0f} ({100*cond_len/orig_len:.0f}%)")

log(f"\n{'='*60}")
log(f"🏆 DONE! Submit in this order:")
log(f"  1. {fname_c} (full hybrid — highest potential)")
log(f"  2. {fname_a} (condensed RL only — test ROUGE-L improvement)")
log(f"  3. {fname_b} (polished LLM only — test LLM improvement)")
log(f"{'='*60}")
