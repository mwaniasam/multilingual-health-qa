#!/usr/bin/env python3
"""Local evaluation harness for the Multilingual Health QA competition.

Computes ROUGE-1 F1, ROUGE-L F1, and a combined score on the validation set
to match the leaderboard metric weighting:
  - ROUGE-1 F1: 37%
  - ROUGE-L F1: 37%
  - LLM-as-Judge: 26% (approximated locally as average of R1 and RL)

Usage:
    python evaluate_local.py --predictions path/to/predictions.csv
    python evaluate_local.py --pred-col TargetRLF1 --predictions path/to/preds.csv
"""
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer


class WhitespaceTokenizer:
    """Language-agnostic whitespace tokenizer for ROUGE scoring."""
    def tokenize(self, text):
        if text is None:
            return []
        return str(text).strip().split()


def compute_rouge_scores(predictions, references):
    """Compute ROUGE-1 and ROUGE-L F1 scores.
    
    Args:
        predictions: list of predicted answer strings
        references: list of reference answer strings
    
    Returns:
        dict with rouge1, rougeL, and combined scores
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rougeL'],
        use_stemmer=False,
        tokenizer=WhitespaceTokenizer()
    )
    
    rouge1_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        pred = str(pred).strip() if pd.notna(pred) else ""
        ref = str(ref).strip() if pd.notna(ref) else ""
        
        if not ref:
            continue
            
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    r1 = np.mean(rouge1_scores)
    rl = np.mean(rougeL_scores)
    # Approximate combined score (LLM-judge ≈ avg of R1 and RL as proxy)
    combined = 0.37 * r1 + 0.37 * rl + 0.26 * ((r1 + rl) / 2)
    
    return {
        'rouge1_f1': r1,
        'rougeL_f1': rl,
        'combined_approx': combined,
        'n_samples': len(rouge1_scores)
    }


def compute_rouge_per_language(predictions_df, references_df, pred_col='prediction', ref_col='output', lang_col='subset'):
    """Compute ROUGE scores broken down by language/subset."""
    merged = predictions_df.merge(references_df[['ID', ref_col, lang_col]], on='ID', how='inner')
    
    results = {}
    for lang in sorted(merged[lang_col].unique()):
        mask = merged[lang_col] == lang
        lang_preds = merged.loc[mask, pred_col].tolist()
        lang_refs = merged.loc[mask, ref_col].tolist()
        scores = compute_rouge_scores(lang_preds, lang_refs)
        scores['count'] = int(mask.sum())
        results[lang] = scores
    
    # Overall
    overall = compute_rouge_scores(merged[pred_col].tolist(), merged[ref_col].tolist())
    overall['count'] = len(merged)
    results['OVERALL'] = overall
    
    return results


def print_results(results, title="Evaluation Results"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(f"{'Subset':<15} {'Count':>6} {'ROUGE-1':>10} {'ROUGE-L':>10} {'Combined':>10}")
    print(f"{'-'*70}")
    
    for lang, scores in sorted(results.items()):
        if lang == 'OVERALL':
            continue
        print(f"{lang:<15} {scores['count']:>6} {scores['rouge1_f1']:>10.4f} {scores['rougeL_f1']:>10.4f} {scores['combined_approx']:>10.4f}")
    
    if 'OVERALL' in results:
        print(f"{'-'*70}")
        overall = results['OVERALL']
        print(f"{'OVERALL':<15} {overall['count']:>6} {overall['rouge1_f1']:>10.4f} {overall['rougeL_f1']:>10.4f} {overall['combined_approx']:>10.4f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Local ROUGE evaluation')
    parser.add_argument('--predictions', required=True, help='Path to predictions CSV')
    parser.add_argument('--val-path', default='data/raw/Val.csv', help='Path to validation CSV')
    parser.add_argument('--pred-col', default='prediction', help='Column name for predictions')
    args = parser.parse_args()
    
    val_df = pd.read_csv(args.val_path)
    pred_df = pd.read_csv(args.predictions)
    
    if args.pred_col not in pred_df.columns:
        # Try common column names
        for col in ['prediction', 'TargetRLF1', 'TargetR1F1', 'TargetLLM', 'output']:
            if col in pred_df.columns:
                args.pred_col = col
                break
    
    results = compute_rouge_per_language(pred_df, val_df, pred_col=args.pred_col)
    print_results(results)


if __name__ == '__main__':
    main()
