#!/usr/bin/env python3
"""Semantic retrieval system using multilingual embeddings.

Uses intfloat/multilingual-e5-base for dense retrieval across African languages.
Builds FAISS index from train+val data for efficient nearest-neighbor search.
"""
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
INDEX_PATH = PROCESSED_DIR / 'faiss_index.bin'
META_PATH = PROCESSED_DIR / 'retrieval_meta.pkl'
EMBEDDINGS_PATH = PROCESSED_DIR / 'embeddings.npy'

MODEL_NAME = 'intfloat/multilingual-e5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SemanticRetriever:
    """Dense retrieval using multilingual-e5-base embeddings + FAISS."""

    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.index = None           # FAISS index
        self.metadata = None        # list of dicts: {id, question, answer, subset}
        self.subset_indices = {}    # subset -> list of indices into metadata

    def _load_model(self):
        """Load the embedding model (lazy init)."""
        if self.model is not None:
            return
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def encode_texts(self, texts, prefix='query: ', batch_size=32, show_progress=True):
        """Encode texts to normalized embeddings.

        Args:
            texts: list of strings to encode
            prefix: 'query: ' for questions, 'passage: ' for answers/passages
            batch_size: encoding batch size
            show_progress: show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), embed_dim), L2-normalized
        """
        self._load_model()

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Encoding ({prefix.strip()})", unit="batch")

        for i in iterator:
            batch = [prefix + t for t in texts[i:i + batch_size]]
            encoded = self.tokenizer(
                batch,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            outputs = self.model(**encoded)
            # Mean pooling over attention mask
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def build_index(self, train_path=None, val_path=None, save=True):
        """Build FAISS index from train + val data.

        We index the QUESTIONS (not answers) so we can find similar questions
        and return their answers.
        """
        train_path = train_path or RAW_DIR / 'Train.csv'
        val_path = val_path or RAW_DIR / 'Val.csv'

        print("Loading data...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Combine train + val (val has reference answers we can use as context)
        combined = pd.concat([train_df, val_df], ignore_index=True)
        combined = combined.dropna(subset=['input', 'output'])
        print(f"Combined dataset: {len(combined)} entries")

        # Build metadata
        self.metadata = []
        for _, row in combined.iterrows():
            self.metadata.append({
                'id': row['ID'],
                'question': str(row['input']).strip(),
                'answer': str(row['output']).strip(),
                'subset': row['subset'],
            })

        # Build subset index
        self.subset_indices = {}
        for idx, meta in enumerate(self.metadata):
            subset = meta['subset']
            if subset not in self.subset_indices:
                self.subset_indices[idx] = []
            self.subset_indices.setdefault(subset, [])
            self.subset_indices[subset].append(idx)

        # Encode all questions
        questions = [m['question'] for m in self.metadata]
        print(f"Encoding {len(questions)} questions...")
        embeddings = self.encode_texts(questions, prefix='query: ', batch_size=64)
        print(f"Embeddings shape: {embeddings.shape}")

        # Build FAISS index (inner product = cosine sim since L2-normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        if save:
            self._save(embeddings)

        return self

    def _save(self, embeddings=None):
        """Save index and metadata to disk."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'subset_indices': self.subset_indices,
            }, f)
        if embeddings is not None:
            np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Saved index to {INDEX_PATH}")
        print(f"Saved metadata to {META_PATH}")

    def load(self):
        """Load pre-built index and metadata from disk."""
        print(f"Loading FAISS index from {INDEX_PATH}...")
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.subset_indices = data['subset_indices']
        print(f"Loaded: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")
        return self

    def retrieve_top_k(self, query, subset=None, k=5, cross_lingual_k=3):
        """Retrieve top-K similar Q/A pairs for a query.

        Args:
            query: question text
            subset: language/subset code (e.g., 'Eng_Uga')
            k: number of same-language results to return
            cross_lingual_k: number of cross-language results to add

        Returns:
            list of dicts with keys: question, answer, subset, score, source
        """
        self._load_model()

        # Encode query
        query_embedding = self.encode_texts([query], prefix='query: ', show_progress=False)

        # Search the full index
        total_k = min(k + cross_lingual_k + 20, self.index.ntotal)  # fetch extra to filter
        scores, indices = self.index.search(query_embedding.astype(np.float32), total_k)

        results = []
        same_lang_count = 0
        cross_lang_count = 0

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]

            if subset and meta['subset'] == subset:
                if same_lang_count < k:
                    results.append({
                        'question': meta['question'],
                        'answer': meta['answer'],
                        'subset': meta['subset'],
                        'score': float(score),
                        'source': 'same_language',
                    })
                    same_lang_count += 1
            elif cross_lang_count < cross_lingual_k:
                results.append({
                    'question': meta['question'],
                    'answer': meta['answer'],
                    'subset': meta['subset'],
                    'score': float(score),
                    'source': 'cross_language',
                })
                cross_lang_count += 1

            if same_lang_count >= k and cross_lang_count >= cross_lingual_k:
                break

        return results

    def retrieve_best_answer(self, query, subset=None):
        """Retrieve the single best matching answer (for ROUGE columns).

        Prioritizes same-language matches.
        """
        results = self.retrieve_top_k(query, subset=subset, k=3, cross_lingual_k=2)
        # Return best same-language match, or best overall
        same_lang = [r for r in results if r['source'] == 'same_language']
        if same_lang:
            return same_lang[0]['answer']
        return results[0]['answer'] if results else ""


def main():
    """Build the semantic retrieval index."""
    print("="*60)
    print("BUILDING SEMANTIC RETRIEVAL INDEX")
    print("="*60)

    retriever = SemanticRetriever()
    retriever.build_index()

    # Quick sanity check
    print("\n" + "="*60)
    print("SANITY CHECK")
    print("="*60)

    test_queries = [
        ("Treatment for Gonorrhea?", "Eng_Uga"),
        ("Dɛn ne aduru a wodi si nyisɛn ano ntɛm ntɛm?", "Aka_Gha"),
        ("What is HIV?", "Eng_Ken"),
    ]

    for query, subset in test_queries:
        results = retriever.retrieve_top_k(query, subset=subset, k=2, cross_lingual_k=1)
        print(f"\nQuery [{subset}]: {query[:60]}...")
        for i, r in enumerate(results):
            print(f"  [{r['source'][:4]}] [{r['subset']}] score={r['score']:.3f}: Q={r['question'][:50]}...")
            print(f"         A={r['answer'][:80]}...")

    print("\n✅ Semantic retrieval index built successfully!")


if __name__ == '__main__':
    main()
