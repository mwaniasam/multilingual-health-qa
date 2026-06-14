# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mwaniasam/multilingual-health-qa/blob/main/notebooks/full_pipeline.ipynb)
[![Zindi Competition](https://img.shields.io/badge/Zindi-Competition-blue)](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages)

> **Course:** Machine Learning Techniques I — Final Course Project (40%)  
> **Author:** Samuel Mwania  
> **Zindi Username:** mwaniasam  
> **Final Public Score:** 0.6908 (V7) · Rank ~13–15 / 200+  
> **Score Progression:** 0.4908 → 0.6908 (+41% relative improvement)

## Overview

A retrieval-and-selection pipeline for multilingual health question answering across **8 language–country subsets** spanning **5 African languages**: Akan (Ghana), Amharic (Ethiopia), English (Ethiopia, Ghana, Kenya, Uganda), Luganda (Uganda), and Swahili (Kenya).

The system retrieves candidate answers from a 36,501-pair training corpus using dense semantic search, then applies per-language selection strategies (MBR consensus, extractive stitching, cross-encoder reranking, or fine-tuned generation) to maximize ROUGE and LLM-as-Judge scores.

## Final Architecture (V7)

```
Test Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│  AfriE5-Large-instruct (McGill-NLP)             │
│  Fine-tuned Q→Q with MNRL + hard negatives      │
│  + Embedding interpolation (β·AfriE5 + (1-β)·FT2)│
└──────────────────┬──────────────────────────────┘
                   │ Top-15 candidates
                   ▼
┌─────────────────────────────────────────────────┐
│  Per-Language Routing                           │
│                                                 │
│  Strong langs (Uga/Ken/Swa) → MBR top-1         │
│  Aka_Gha, Amh_Eth → CE reranker → stitch        │
│  Eng_Gha → Qwen2.5-7B LoRA generation           │
│  Lug_Uga → Per-lang adapter (β=0.8) + MBR       │
└─────────────────────────────────────────────────┘
                   │
                   ▼
         Identical-column submission
```

**All components are fully open-source.** No paid APIs in the final solution.

## Score Progression

| # | Experiment | Public Score | Δ | Key Change |
|---|-----------|:-----------:|:---:|-----------|
| 1 | TF-IDF char n-gram retrieval | 0.4908 | — | Baseline |
| 2 | mT5-base seq2seq | 0.2396 | −0.251 | Generation destroys ROUGE |
| 3 | AfriTeVa V2 seq2seq | 0.2971 | — | Better than mT5, still below retrieval |
| 4 | E5-base semantic retrieval | 0.5742 | +0.083 | Dense embeddings leap |
| 5 | AfriE5-Large + HN mining | 0.6545 | +0.080 | SOTA African embedder |
| 6 | + MBR consensus selection | 0.6597 | +0.005 | Consensus over top-1 |
| 7 | + Extractive stitcher | 0.6650 | +0.005 | Verbatim sentence extraction |
| 8 | + Unicode fix + identical cols | 0.6843 | +0.019 | Correct Amharic/Akan scoring |
| 9 | + Embedding interpolation | **0.6898** | +0.006 | **V4 (selected final)** |
| 10 | + CE reranker + QA-union | **0.6908** | +0.001 | **V7 (BEST, selected final)** |

Full experiment ledger (44 experiments): [`docs/experiment_log.csv`](docs/experiment_log.csv)

## Project Structure

```
multilingual-health-qa/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/                          # Competition data
│       ├── Train.csv                 # 36,501 Q/A pairs (29,815 train + 6,686 val)
│       ├── Test.csv                  # 2,618 test questions
│       └── SampleSubmission.csv
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb     # EDA + TF-IDF baseline (Exp 1)
│   ├── 02_mt5base_finetuning.ipynb   # mT5-base seq2seq (Exp 2)
│   ├── 03_afriteva_finetuning.ipynb  # AfriTeVa V2 (Exp 3)
│   ├── full_pipeline.ipynb           # End-to-end Colab-ready pipeline
│   └── experiments/                  # All experiment scripts (.py)
├── src/
│   ├── bootstrap_contract.py         # Pipeline helper functions + data contract
│   ├── training_cells.py             # CE reranker, FT2, Lug adapter, Qwen LoRA
│   ├── build_submissions.py          # V4, V6, V7 submission builders
│   ├── semantic_retrieval.py         # Dense retrieval with E5 embeddings
│   ├── evaluate_local.py             # Local ROUGE evaluation harness
│   └── ...
├── submissions/                      # All submission CSVs
│   ├── submission_v7.csv             # BEST (0.6908) — selected final
│   ├── submission_v4_final.csv       # Hedge (0.6898) — selected final
│   └── ...                           # Earlier experiments
└── docs/
    ├── SamuelMwania_FinalProject.md  # Academic report
    ├── FINDINGS_AND_HANDOFF.md       # Complete findings and architecture record
    ├── experiment_log.csv            # All 44 experiments with verdicts
    └── ARTIFACTS_MANIFEST.md         # Drive artifact checklist
```

## Quick Start (Google Colab)

1. Click the **Open in Colab** badge above
2. Run the bootstrap cell (loads data, builds embeddings, defines helpers)
3. Run training cells to load pre-trained models from Drive
4. Run the V7 builder to generate the submission

**Models on Google Drive** (too large for GitHub):
- `afrie5-final-model/` — retrieval backbone
- `ce-reranker-v2/` — cross-encoder (xlm-roberta-base)
- `qwen-ft-health/` — Qwen2.5-7B LoRA adapter
- `afrie5-lug_uga/` — Luganda per-language adapter

See [`docs/ARTIFACTS_MANIFEST.md`](docs/ARTIFACTS_MANIFEST.md) for the complete artifact checklist.

## Technologies

| Component | Model/Tool | Purpose |
|-----------|-----------|---------|
| Retrieval backbone | [AfriE5-Large-instruct](https://huggingface.co/McGill-NLP/AfriE5-large-instruct) | Dense Q→Q semantic search |
| Cross-encoder | xlm-roberta-base (binary classification) | Rerank candidates for Aka/Amh |
| Generation | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (LoRA r=16) | Eng_Gha answer generation |
| Search index | FAISS (IndexFlatIP) | Fast cosine similarity search |
| Selection | MBR consensus + extractive stitcher | Per-language answer optimization |
| Evaluation | Unicode ROUGE tokenizer (re.UNICODE) | Matches organizer's fixed scorer |

## Evaluation Metrics

The competition uses a weighted combination:
- **ROUGE-1 F1** (37%): Unigram overlap
- **ROUGE-L F1** (37%): Longest common subsequence
- **LLM-as-a-Judge** (26%): Factual accuracy, completeness, language quality

## Ethical Considerations

- **Health misinformation risk:** All outputs are grounded in retrieved training data curated by health professionals. No free-form generation on strong languages.
- **Language equity:** This work addresses the critical gap in NLP tools for low-resource African languages.
- **Cultural sensitivity:** Health topics around sexuality and reproduction require cultural awareness. The system respects local terminology.
- **Bias in evaluation:** ROUGE metrics disadvantage morphologically rich languages (Akan, Amharic) due to tokenization differences.

## AI Usage Disclosure

- **Claude (Anthropic)** and **Antigravity (Google)** were used as coding assistants for experimentation, debugging, and code generation.
- **Gemini API** was used in early experiments (Exp 6–11) but was **removed from the final solution** for open-source compliance.
- All architectural decisions, experiment design, and analysis reflect the author's own understanding and judgment.

## Acknowledgments

- [Zindi Africa](https://zindi.africa) for hosting the competition
- [ITU](https://www.itu.int/), HASH, and Makerere University for organizing the challenge
- [McGill-NLP](https://github.com/McGill-NLP) for AfriE5-Large-instruct
- Health professionals who curated the multilingual Q/A dataset