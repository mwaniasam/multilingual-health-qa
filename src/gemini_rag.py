#!/usr/bin/env python3
"""Gemini RAG (Retrieval-Augmented Generation) for multilingual health QA.

Uses the Gemini API to generate high-quality answers grounded in retrieved context.
Produces separate answers optimized for different evaluation metrics.
"""
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types as genai_types

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# ── Language mapping ──────────────────────────────────────────────────────────
SUBSET_TO_LANG = {
    'Aka_Gha': 'Akan (Ghana)',
    'Amh_Eth': 'Amharic (Ethiopia)',
    'Eng_Eth': 'English (Ethiopia)',
    'Eng_Gha': 'English (Ghana)',
    'Eng_Ken': 'English (Kenya)',
    'Eng_Uga': 'English (Uganda)',
    'Lug_Uga': 'Luganda (Uganda)',
    'Swa_Ken': 'Swahili (Kenya)',
}

SUBSET_TO_SCRIPT_HINT = {
    'Aka_Gha': 'Write entirely in Akan (Twi). Use Akan vocabulary and grammar.',
    'Amh_Eth': 'Write entirely in Amharic using the Ge\'ez/Ethiopic script (fidäl). Do NOT use Latin characters.',
    'Eng_Eth': 'Write in English, using terminology appropriate for Ethiopian health context.',
    'Eng_Gha': 'Write in English, using terminology appropriate for Ghanaian health context.',
    'Eng_Ken': 'Write in English, using terminology appropriate for Kenyan health context.',
    'Eng_Uga': 'Write in English, using simple medical terms appropriate for Ugandan context.',
    'Lug_Uga': 'Write entirely in Luganda. Use Luganda vocabulary and grammar.',
    'Swa_Ken': 'Write entirely in Swahili (Kiswahili). Use Swahili vocabulary and grammar.',
}


class GeminiRAG:
    """Gemini-powered answer generation with retrieval augmentation."""

    def __init__(self, api_key=None, model='gemini-2.5-flash'):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Pass api_key or set env var.")
        self.model_name = model
        self.client = genai.Client(api_key=self.api_key)
        self._call_count = 0
        self._last_call_time = 0

    def _rate_limit(self):
        """Enforce rate limiting (paid tier: ~1000 RPM, use 0.5s gap for safety)."""
        elapsed = time.time() - self._last_call_time
        wait = max(0, 0.5 - elapsed)  # Paid tier allows much higher throughput
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.time()
        self._call_count += 1

    def _call_gemini(self, prompt, temperature=0.3, max_tokens=512, retries=3):
        """Call Gemini API with retries and rate limiting."""
        for attempt in range(retries):
            try:
                self._rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                if response.text:
                    return response.text.strip()
                return ""
            except Exception as e:
                wait_time = (2 ** attempt) * 5
                print(f"  [WARN] API error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [ERROR] All retries failed for this query.")
                    traceback.print_exc()
                    return ""

    def _format_context(self, retrieved_contexts, max_contexts=5):
        """Format retrieved Q/A pairs into a context string."""
        lines = []
        for i, ctx in enumerate(retrieved_contexts[:max_contexts], 1):
            lang = SUBSET_TO_LANG.get(ctx.get('subset', ''), ctx.get('subset', 'Unknown'))
            lines.append(f"--- Reference {i} [{lang}] ---")
            lines.append(f"Question: {ctx['question']}")
            lines.append(f"Answer: {ctx['answer']}")
            lines.append("")
        return "\n".join(lines)

    def generate_for_llm_judge(self, question, subset, retrieved_contexts):
        """Generate answer optimized for LLM-as-Judge evaluation.

        Prioritizes: factual accuracy, completeness, language appropriateness,
        cultural relevance, and natural fluency.
        """
        language = SUBSET_TO_LANG.get(subset, subset)
        script_hint = SUBSET_TO_SCRIPT_HINT.get(subset, f'Write in {language}.')
        context_str = self._format_context(retrieved_contexts)

        prompt = f"""You are an expert health educator providing answers about maternal, sexual, and reproductive health (MSRH) in {language}.

CRITICAL LANGUAGE RULE: {script_hint}

Your answer will be evaluated by an AI judge on these criteria:
1. FACTUAL ACCURACY: Every medical claim must be correct
2. COMPLETENESS: Address all aspects of the question thoroughly
3. LANGUAGE QUALITY: Natural, fluent {language} — not a translation
4. CULTURAL APPROPRIATENESS: Sensitive to local health context
5. HELPFULNESS: Practical, actionable information

REFERENCE INFORMATION (use these for factual grounding):
{context_str}

QUESTION: {question}

Provide a comprehensive, accurate answer in {language}. Be thorough but concise. Use the reference information for factual grounding but write naturally — do NOT copy references verbatim. The answer must sound like it was originally written in {language} by a native speaker health expert.

ANSWER:"""

        return self._call_gemini(prompt, temperature=0.3, max_tokens=600)

    def generate_for_rouge(self, question, subset, retrieved_contexts):
        """Generate answer optimized for ROUGE-L/ROUGE-1 overlap.

        Prioritizes: reusing exact phrasing, including all key medical terms,
        matching reference answer structure and vocabulary.
        """
        language = SUBSET_TO_LANG.get(subset, subset)
        script_hint = SUBSET_TO_SCRIPT_HINT.get(subset, f'Write in {language}.')

        # For ROUGE, same-language contexts are most valuable
        same_lang = [c for c in retrieved_contexts if c.get('subset') == subset]
        other_lang = [c for c in retrieved_contexts if c.get('subset') != subset]
        ordered_contexts = same_lang + other_lang
        context_str = self._format_context(ordered_contexts)

        prompt = f"""You are answering a health question in {language}. 

CRITICAL: {script_hint}

Your answer will be evaluated by measuring WORD OVERLAP with a reference answer. To score well:
- REUSE exact phrases, sentences, and medical terms from the reference answers below
- Include ALL key medical terminology mentioned in the references
- Match the STYLE, STRUCTURE, and LENGTH of the reference answers
- Incorporate as many exact word sequences from the references as possible
- If same-language references exist, prefer copying their phrasing

REFERENCE ANSWERS:
{context_str}

QUESTION: {question}

Write your answer in {language}. MAXIMIZE word overlap with the references above. Use their exact phrasing wherever possible. Combine information from multiple references if relevant.

ANSWER:"""

        return self._call_gemini(prompt, temperature=0.1, max_tokens=600)

    def batch_generate(self, test_df, retrieval_fn, mode='llm',
                       progress_path=None, save_every=50):
        """Generate answers for entire test set with progress saving.

        Args:
            test_df: DataFrame with ID, input, subset columns
            retrieval_fn: callable(question, subset) -> list of context dicts
            mode: 'llm' for LLM-judge optimization, 'rouge' for ROUGE optimization
            progress_path: path to save progress JSON
            save_every: save progress every N questions

        Returns:
            dict of {ID: generated_answer}
        """
        if progress_path is None:
            progress_path = PROCESSED_DIR / f'gemini_progress_{mode}.json'
        progress_path = Path(progress_path)

        # Load existing progress
        results = {}
        if progress_path.exists():
            with open(progress_path) as f:
                results = json.load(f)
            print(f"Resumed from {progress_path}: {len(results)} existing results")

        generate_fn = self.generate_for_llm_judge if mode == 'llm' else self.generate_for_rouge
        total = len(test_df)
        skipped = 0

        print(f"\n{'='*60}")
        print(f"BATCH GENERATION: mode={mode}, total={total}")
        print(f"{'='*60}")

        for i, (_, row) in enumerate(test_df.iterrows()):
            row_id = row['ID']
            if row_id in results:
                skipped += 1
                continue

            question = str(row['input']).strip()
            subset = row['subset']

            # Retrieve context
            contexts = retrieval_fn(question, subset)

            # Generate
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] [{i+1}/{total}] [{subset}] {question[:50]}...")

            answer = generate_fn(question, subset, contexts)
            results[row_id] = answer

            if answer:
                print(f"  -> {answer[:80]}...")
            else:
                print(f"  -> [EMPTY - generation failed]")

            # Save progress
            if (i + 1 - skipped) % save_every == 0:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_path, 'w') as f:
                    json.dump(results, f, ensure_ascii=False)
                print(f"  [Progress saved: {len(results)}/{total}]")

        # Final save
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"\n✅ Generation complete: {len(results)} answers saved to {progress_path}")
        if skipped > 0:
            print(f"   (Skipped {skipped} already-generated answers)")

        return results


def main():
    """Quick test of the Gemini RAG system."""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first!")
        return

    rag = GeminiRAG(api_key=api_key)

    # Test with a simple query
    test_contexts = [
        {
            'question': 'What is HIV?',
            'answer': 'HIV (Human Immunodeficiency Virus) is a virus that attacks the immune system. It destroys CD4 cells which help fight infections.',
            'subset': 'Eng_Uga',
        }
    ]

    print("Testing LLM-judge generation...")
    result_llm = rag.generate_for_llm_judge(
        "How does HIV spread?",
        "Eng_Uga",
        test_contexts
    )
    print(f"LLM result: {result_llm[:200]}")

    print("\nTesting ROUGE generation...")
    result_rouge = rag.generate_for_rouge(
        "How does HIV spread?",
        "Eng_Uga",
        test_contexts
    )
    print(f"ROUGE result: {result_rouge[:200]}")

    print("\n✅ Gemini RAG test complete!")


if __name__ == '__main__':
    main()
