#!/usr/bin/env python3
"""Main pipeline: Semantic Retrieval + Gemini RAG + Per-Metric Optimization.

Orchestrates the full competition pipeline:
1. Build/load semantic retrieval index
2. Generate answers using Gemini API (separate for ROUGE and LLM-judge)
3. Combine with retrieval for per-metric column optimization
4. Produce final submission CSV

Usage:
    # Step 1: Build the retrieval index (run once, ~20 min)
    python run_pipeline.py --step index

    # Step 2: Generate LLM answers via Gemini (run once, ~3 hours for 2618 questions)
    python run_pipeline.py --step generate --api-key YOUR_KEY

    # Step 3: Build the submission
    python run_pipeline.py --step submit --api-key YOUR_KEY

    # All steps at once:
    python run_pipeline.py --step all --api-key YOUR_KEY
"""
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_retrieval import SemanticRetriever
from gemini_rag import GeminiRAG
from evaluate_local import compute_rouge_per_language, print_results

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SUBMISSIONS_DIR = PROJECT_DIR / 'submissions'


def step_build_index():
    """Step 1: Build the semantic retrieval index."""
    print("\n" + "="*70)
    print(" STEP 1: BUILD SEMANTIC RETRIEVAL INDEX")
    print("="*70)

    retriever = SemanticRetriever()
    retriever.build_index()
    return retriever


def step_generate(api_key, retriever=None):
    """Step 2: Generate answers using Gemini API."""
    print("\n" + "="*70)
    print(" STEP 2: GENERATE ANSWERS WITH GEMINI")
    print("="*70)

    # Load retriever if not provided
    if retriever is None:
        retriever = SemanticRetriever()
        retriever.load()

    # Load test data
    test_df = pd.read_csv(RAW_DIR / 'Test.csv')
    print(f"Test set: {len(test_df)} questions")

    # Initialize Gemini RAG
    rag = GeminiRAG(api_key=api_key)

    # Retrieval function for the RAG pipeline
    def retrieval_fn(question, subset):
        return retriever.retrieve_top_k(question, subset=subset, k=5, cross_lingual_k=3)

    # Generate ROUGE-optimized answers
    print("\n--- Generating ROUGE-optimized answers ---")
    rouge_results = rag.batch_generate(
        test_df, retrieval_fn, mode='rouge',
        progress_path=PROCESSED_DIR / 'gemini_progress_rouge.json'
    )

    # Generate LLM-judge-optimized answers
    print("\n--- Generating LLM-judge-optimized answers ---")
    llm_results = rag.batch_generate(
        test_df, retrieval_fn, mode='llm',
        progress_path=PROCESSED_DIR / 'gemini_progress_llm.json'
    )

    return rouge_results, llm_results


def step_submit(api_key=None, retriever=None, validate=True):
    """Step 3: Build the optimized submission."""
    print("\n" + "="*70)
    print(" STEP 3: BUILD OPTIMIZED SUBMISSION")
    print("="*70)

    # Load retriever
    if retriever is None:
        retriever = SemanticRetriever()
        retriever.load()

    # Load test data
    test_df = pd.read_csv(RAW_DIR / 'Test.csv')

    # Load generated answers
    rouge_path = PROCESSED_DIR / 'gemini_progress_rouge.json'
    llm_path = PROCESSED_DIR / 'gemini_progress_llm.json'

    rouge_results = {}
    llm_results = {}

    if rouge_path.exists():
        with open(rouge_path) as f:
            rouge_results = json.load(f)
        print(f"Loaded {len(rouge_results)} ROUGE-optimized answers")
    else:
        print("[WARN] No ROUGE results found. Will use retrieval-only.")

    if llm_path.exists():
        with open(llm_path) as f:
            llm_results = json.load(f)
        print(f"Loaded {len(llm_results)} LLM-optimized answers")
    else:
        print("[WARN] No LLM results found. Will use retrieval-only.")

    # Build submission with per-metric optimization
    submission_rows = []

    for _, row in test_df.iterrows():
        row_id = row['ID']
        question = str(row['input']).strip()
        subset = row['subset']

        # Get retrieval-based answer (best for ROUGE when good match exists)
        retrieved_answer = retriever.retrieve_best_answer(question, subset=subset)

        # Get Gemini-generated answers
        gemini_rouge = rouge_results.get(row_id, '')
        gemini_llm = llm_results.get(row_id, '')

        # PER-METRIC COLUMN OPTIMIZATION:
        #
        # TargetRLF1 (ROUGE-L): Use ROUGE-optimized Gemini answer if available,
        #   fallback to retrieved answer. Gemini ROUGE prompts incorporate verbatim
        #   phrasing but also answer the ACTUAL question (not the retrieved one).
        #
        # TargetR1F1 (ROUGE-1): Same strategy — unigram overlap benefits from
        #   including all relevant medical terms.
        #
        # TargetLLM (LLM-judge): Use LLM-optimized Gemini answer for quality,
        #   fallback to retrieved answer.

        target_rlf1 = gemini_rouge if gemini_rouge else retrieved_answer
        target_r1f1 = gemini_rouge if gemini_rouge else retrieved_answer
        target_llm = gemini_llm if gemini_llm else retrieved_answer

        # Ensure no empty answers (submission requirement)
        target_rlf1 = target_rlf1 if target_rlf1.strip() else retrieved_answer
        target_r1f1 = target_r1f1 if target_r1f1.strip() else retrieved_answer
        target_llm = target_llm if target_llm.strip() else retrieved_answer

        # Final fallback
        fallback = "This is a health-related question that requires professional medical advice."
        target_rlf1 = target_rlf1 if target_rlf1.strip() else fallback
        target_r1f1 = target_r1f1 if target_r1f1.strip() else fallback
        target_llm = target_llm if target_llm.strip() else fallback

        submission_rows.append({
            'ID': row_id,
            'TargetRLF1': target_rlf1.strip(),
            'TargetR1F1': target_r1f1.strip(),
            'TargetLLM': target_llm.strip(),
        })

    submission_df = pd.DataFrame(submission_rows)

    # Verify submission format
    sample_sub = pd.read_csv(RAW_DIR / 'SampleSubmission.csv')
    assert list(submission_df.columns) == list(sample_sub.columns), \
        f"Column mismatch: {submission_df.columns.tolist()} vs {sample_sub.columns.tolist()}"
    assert len(submission_df) == len(sample_sub), \
        f"Row count mismatch: {len(submission_df)} vs {len(sample_sub)}"
    assert submission_df['ID'].tolist() == sample_sub['ID'].tolist(), \
        "ID order mismatch!"

    # Check for empty answers
    for col in ['TargetRLF1', 'TargetR1F1', 'TargetLLM']:
        empty = submission_df[col].str.strip().eq('').sum()
        if empty > 0:
            print(f"[WARN] {empty} empty values in {col}")

    # Save submission
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMISSIONS_DIR / f'submission_semantic_rag_{timestamp}.csv'
    submission_df.to_csv(sub_path, index=False)
    print(f"\n✅ Submission saved to {sub_path}")
    print(f"   Shape: {submission_df.shape}")

    # Also save as "latest"
    latest_path = SUBMISSIONS_DIR / 'submission_latest.csv'
    submission_df.to_csv(latest_path, index=False)
    print(f"   Also saved as {latest_path}")

    # Validate on val set if requested
    if validate:
        _validate_on_val(retriever, api_key)

    return submission_df


def _validate_on_val(retriever, api_key=None):
    """Run local ROUGE validation on the val set using retrieval."""
    print("\n" + "="*70)
    print(" LOCAL VALIDATION (Val Set, Retrieval Only)")
    print("="*70)

    val_df = pd.read_csv(RAW_DIR / 'Val.csv')

    # Generate retrieval predictions for val
    predictions = []
    for _, row in val_df.iterrows():
        question = str(row['input']).strip()
        subset = row['subset']
        answer = retriever.retrieve_best_answer(question, subset=subset)
        predictions.append({
            'ID': row['ID'],
            'prediction': answer,
        })

    pred_df = pd.DataFrame(predictions)

    results = compute_rouge_per_language(pred_df, val_df, pred_col='prediction')
    print_results(results, title="Semantic Retrieval on Val Set")

    return results


def main():
    parser = argparse.ArgumentParser(description='Multilingual Health QA Pipeline')
    parser.add_argument('--step', choices=['index', 'generate', 'submit', 'validate', 'all'],
                        default='all', help='Pipeline step to run')
    parser.add_argument('--api-key', default=None, help='Gemini API key')
    args = parser.parse_args()

    if args.api_key:
        os.environ['GEMINI_API_KEY'] = args.api_key

    retriever = None

    if args.step in ('index', 'all'):
        retriever = step_build_index()

    if args.step in ('generate', 'all'):
        if retriever is None:
            retriever = SemanticRetriever()
            retriever.load()
        step_generate(
            api_key=args.api_key or os.environ.get('GEMINI_API_KEY', ''),
            retriever=retriever
        )

    if args.step in ('submit', 'all'):
        step_submit(
            api_key=args.api_key or os.environ.get('GEMINI_API_KEY', ''),
            retriever=retriever,
            validate=True
        )

    if args.step == 'validate':
        if retriever is None:
            retriever = SemanticRetriever()
            retriever.load()
        _validate_on_val(retriever)


if __name__ == '__main__':
    main()
