#!/usr/bin/env python3
"""Ensemble retrieval: combines semantic + TF-IDF for better coverage.

Uses both dense semantic retrieval and sparse TF-IDF matching,
selecting the best answer via a confidence heuristic.
"""
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_retrieval import SemanticRetriever

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / 'data' / 'raw'
SUBMISSIONS_DIR = PROJECT_DIR / 'submissions'


class EnsembleRetriever:
    """Combines semantic embeddings + TF-IDF for robust retrieval."""

    def __init__(self):
        self.semantic = SemanticRetriever()
        self.tfidf_indexes = {}

    def build(self):
        """Build both retrieval indexes."""
        # Load semantic
        self.semantic.load()

        # Build TF-IDF per-language
        train_df = pd.read_csv(RAW_DIR / 'Train.csv')
        val_df = pd.read_csv(RAW_DIR / 'Val.csv')
        combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])

        print("Building TF-IDF indexes...")
        for subset in combined['subset'].unique():
            sub_data = combined[combined['subset'] == subset].reset_index(drop=True)
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50000)
            matrix = vec.fit_transform(sub_data['input'].fillna(''))
            self.tfidf_indexes[subset] = {
                'vectorizer': vec,
                'matrix': matrix,
                'data': sub_data,
            }
        print(f"TF-IDF indexes built for {len(self.tfidf_indexes)} subsets")

    def retrieve(self, question, subset):
        """Get best answer using ensemble of semantic + TF-IDF.

        Returns the answer from whichever method has higher confidence.
        """
        # Semantic retrieval
        sem_results = self.semantic.retrieve_top_k(question, subset=subset, k=3, cross_lingual_k=2)
        sem_same = [r for r in sem_results if r['source'] == 'same_language']
        sem_ans = sem_same[0]['answer'] if sem_same else (sem_results[0]['answer'] if sem_results else '')
        sem_score = sem_same[0]['score'] if sem_same else (sem_results[0]['score'] if sem_results else 0)

        # TF-IDF retrieval
        tfidf_ans = ''
        tfidf_score = 0
        if subset in self.tfidf_indexes:
            idx = self.tfidf_indexes[subset]
            q_vec = idx['vectorizer'].transform([question])
            sims = cosine_similarity(q_vec, idx['matrix']).flatten()
            best_i = np.argmax(sims)
            tfidf_ans = str(idx['data'].iloc[best_i]['output'])
            tfidf_score = sims[best_i]

        # Decision heuristic:
        # 1. If semantic has very high confidence (>0.97), trust it
        # 2. If TF-IDF has very high confidence (>0.8), prefer it (exact char match)
        # 3. Otherwise prefer semantic (better at understanding meaning)
        if sem_score > 0.97:
            return sem_ans
        elif tfidf_score > 0.8:
            return tfidf_ans
        elif sem_score > 0.90:
            return sem_ans
        elif tfidf_score > 0.5:
            # Both moderate — pick longer answer (usually more complete)
            return sem_ans if len(sem_ans) >= len(tfidf_ans) else tfidf_ans
        else:
            return sem_ans


def main():
    print("=" * 70)
    print(" ENSEMBLE RETRIEVAL SUBMISSION")
    print("=" * 70)

    ensemble = EnsembleRetriever()
    ensemble.build()

    test_df = pd.read_csv(RAW_DIR / 'Test.csv')
    sample_sub = pd.read_csv(RAW_DIR / 'SampleSubmission.csv')

    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Ensemble"):
        question = str(row['input']).strip()
        subset = row['subset']
        answer = ensemble.retrieve(question, subset)
        fallback = "Health information not available."
        answer = answer if answer.strip() else fallback

        rows.append({
            'ID': row['ID'],
            'TargetRLF1': answer,
            'TargetR1F1': answer,
            'TargetLLM': answer,
        })

    sub = pd.DataFrame(rows)
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = SUBMISSIONS_DIR / f'submission_ensemble_{ts}.csv'
    sub.to_csv(path, index=False)
    sub.to_csv(SUBMISSIONS_DIR / 'submission_latest.csv', index=False)
    print(f"\n✅ Saved: {path}")
    print(f"Shape: {sub.shape}")


if __name__ == '__main__':
    main()
