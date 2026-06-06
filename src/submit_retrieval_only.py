#!/usr/bin/env python3
"""Generate a retrieval-only submission using semantic embeddings.

This is the quick-win baseline: use semantic retrieval instead of TF-IDF.
No LLM generation needed. Can be submitted immediately.
"""
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_retrieval import SemanticRetriever

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / 'data' / 'raw'
SUBMISSIONS_DIR = PROJECT_DIR / 'submissions'


def main():
    print("="*70)
    print(" SEMANTIC RETRIEVAL-ONLY SUBMISSION")
    print("="*70)

    # Load retriever
    retriever = SemanticRetriever()
    retriever.load()

    # Load test data
    test_df = pd.read_csv(RAW_DIR / 'Test.csv')
    sample_sub = pd.read_csv(RAW_DIR / 'SampleSubmission.csv')
    print(f"Test set: {len(test_df)} questions")

    # Generate predictions
    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Retrieving"):
        question = str(row['input']).strip()
        subset = row['subset']

        # Get top results
        results = retriever.retrieve_top_k(question, subset=subset, k=5, cross_lingual_k=3)

        # Best same-language answer (for ROUGE columns)
        same_lang = [r for r in results if r['source'] == 'same_language']
        best_answer = same_lang[0]['answer'] if same_lang else results[0]['answer']

        # For ROUGE-1: concatenate key phrases from top-2 same-lang answers
        # to maximize unigram coverage
        if len(same_lang) >= 2:
            # Use best answer but append unique content from 2nd best
            second_answer = same_lang[1]['answer']
            rouge1_answer = best_answer
            # Only add second if it's substantially different
            if same_lang[1]['score'] > 0.5:
                rouge1_answer = best_answer
            else:
                rouge1_answer = best_answer
        else:
            rouge1_answer = best_answer

        rows.append({
            'ID': row['ID'],
            'TargetRLF1': best_answer,
            'TargetR1F1': rouge1_answer,
            'TargetLLM': best_answer,  # Same for now; Gemini will replace later
        })

    submission_df = pd.DataFrame(rows)

    # Verify format
    assert list(submission_df.columns) == list(sample_sub.columns)
    assert len(submission_df) == len(sample_sub)
    assert submission_df['ID'].tolist() == sample_sub['ID'].tolist()

    # Check for empty
    for col in ['TargetRLF1', 'TargetR1F1', 'TargetLLM']:
        empty = submission_df[col].str.strip().eq('').sum()
        null = submission_df[col].isna().sum()
        print(f"  {col}: {empty} empty, {null} null")

    # Save
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = SUBMISSIONS_DIR / f'submission_semantic_retrieval_{timestamp}.csv'
    submission_df.to_csv(path, index=False)
    print(f"\n✅ Saved to {path}")

    latest = SUBMISSIONS_DIR / 'submission_semantic_retrieval_latest.csv'
    submission_df.to_csv(latest, index=False)
    print(f"   Also saved to {latest}")

    # Print some examples
    print(f"\n{'='*70}")
    print(" EXAMPLES")
    print(f"{'='*70}")
    for i in [0, 500, 1000, 2000]:
        if i < len(test_df):
            print(f"\n[{test_df.iloc[i]['subset']}] Q: {test_df.iloc[i]['input'][:80]}...")
            print(f"  A: {submission_df.iloc[i]['TargetRLF1'][:100]}...")


if __name__ == '__main__':
    main()
