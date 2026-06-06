# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mwaniasam/multilingual-health-qa/blob/main/notebooks/full_pipeline.ipynb)
[![Zindi Competition](https://img.shields.io/badge/Zindi-Competition-blue)](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages)

> **Course:** Machine Learning Techniques I — Final Course Project (40%)  
> **Author:** Samuel Mwania  
> **Zindi Username:** mwaniasam

## Overview

This project develops a multilingual question-answering system for maternal, sexual, and reproductive health (MSRH) topics across **8 language–country configurations** spanning **5 African languages**: Akan (Ghana), Amharic (Ethiopia), English (Ethiopia, Ghana, Kenya, Uganda), Luganda (Uganda), and Swahili (Kenya).

The system uses a **Retrieval-Augmented Generation (RAG)** architecture combining:
1. **Dense semantic retrieval** using multilingual embeddings (`intfloat/multilingual-e5-base`)
2. **LLM-powered answer generation** using Google Gemini (`gemini-2.5-flash`)
3. **Per-metric optimization** — separate answer strategies for ROUGE-1, ROUGE-L, and LLM-as-Judge evaluation

## Project Structure

```
multilingual-health-qa/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                     # Original competition data
│   │   ├── Train.csv            # 29,815 training Q/A pairs
│   │   ├── Val.csv              # 6,686 validation Q/A pairs
│   │   ├── Test.csv             # 2,618 test questions
│   │   └── SampleSubmission.csv # Submission format template
│   └── processed/               # Generated artifacts
│       ├── train_clean.csv      # Cleaned training data
│       ├── faiss_index.bin      # FAISS semantic search index
│       ├── retrieval_meta.pkl   # Retrieval metadata
│       └── embeddings.npy       # Precomputed embeddings
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb         # EDA + TF-IDF baseline
│   ├── 02_mt5base_finetuning.ipynb       # mT5-base fine-tuning
│   ├── 03_afriteva_finetuning.ipynb      # AfriTeVa V2 fine-tuning
│   └── full_pipeline.ipynb               # End-to-end RAG pipeline (Colab-ready)
├── src/
│   ├── __init__.py
│   ├── semantic_retrieval.py    # Dense retrieval with multilingual-e5-base
│   ├── gemini_rag.py            # Gemini RAG generation module
│   ├── evaluate_local.py        # Local ROUGE evaluation harness
│   ├── run_pipeline.py          # Main orchestration script
│   └── submit_retrieval_only.py # Retrieval-only submission generator
├── submissions/                 # Generated submission CSVs
├── docs/
│   ├── experiment_log.csv       # Detailed experiment tracking
│   └── language_distribution.png
└── .gitignore
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM) or Google Colab
- Gemini API key ([get one free](https://ai.google.dev/))

### Installation

```bash
git clone https://github.com/mwaniasam/multilingual-health-qa.git
cd multilingual-health-qa
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: Build the semantic retrieval index (~15 min)
python src/run_pipeline.py --step index

# Step 2: Generate answers using Gemini RAG (~45 min with paid API)
python src/run_pipeline.py --step generate --api-key YOUR_GEMINI_KEY

# Step 3: Build optimized submission
python src/run_pipeline.py --step submit --api-key YOUR_GEMINI_KEY

# Or run everything at once:
python src/run_pipeline.py --step all --api-key YOUR_GEMINI_KEY
```

### Google Colab

Click the Colab badge above or open `notebooks/full_pipeline.ipynb` directly in Colab. The notebook includes all setup steps and can run end-to-end with a free Colab GPU.

## Methodology

### Data Understanding

| Subset | Language | Country | Train | Val | Test |
|--------|----------|---------|-------|-----|------|
| Eng_Uga | English | Uganda | 7,624 | 1,688 | 744 |
| Aka_Gha | Akan | Ghana | 4,455 | 1,114 | 492 |
| Eng_Gha | English | Ghana | 4,443 | 1,104 | 491 |
| Eng_Eth | English | Ethiopia | 3,915 | 564 | 60 |
| Lug_Uga | Luganda | Uganda | 3,383 | 846 | 374 |
| Eng_Ken | English | Kenya | 2,080 | 390 | 167 |
| Swa_Ken | Swahili | Kenya | 2,070 | 518 | 229 |
| Amh_Eth | Amharic | Ethiopia | 1,845 | 462 | 61 |

**Key data insight:** Train and test share topic IDs (hash suffixes), but test questions are always *different* from training questions on the same topic. Furthermore, 81% of test entries require cross-lingual transfer — the training data for that topic exists only in a different language.

### Approach Evolution

| # | Experiment | Approach | Public Score | Key Insight |
|---|-----------|----------|-------------|-------------|
| 1 | TF-IDF Baseline | char_wb (3,5) 1-NN retrieval | 0.4908 | Strong non-neural floor |
| 2 | mT5-base Fine-tuned | Seq2seq generation | 0.2396 | Generation rephrases, hurts ROUGE |
| 3 | Tuned TF-IDF | char_wb (2,4) n-grams | 0.4962 | Shorter n-grams generalize better |
| 4 | AfriTeVa V2 | Africa-centric seq2seq | 0.2971 | +0.057 over mT5, still below retrieval |
| 5 | Semantic Retrieval | multilingual-e5-base + FAISS | *pending* | Dense embeddings capture topic semantics |
| 6 | RAG (ROUGE-opt) | Gemini 2.5 Flash + retrieved context | *pending* | Verbatim phrasing from context |
| 7 | RAG (LLM-opt) | Gemini 2.5 Flash quality generation | *pending* | Optimized for LLM-as-Judge |
| 8 | Per-Metric Split | Different answers per target column | *pending* | Exploits multi-metric evaluation |

### Evaluation Metrics

The competition uses a weighted combination:
- **ROUGE-1 F1** (37%): Unigram overlap
- **ROUGE-L F1** (37%): Longest common subsequence
- **LLM-as-a-Judge** (26%): Factual accuracy, completeness, language quality

Our per-metric optimization strategy submits different answers for each metric column.

## Ethical Considerations

- **Health misinformation risk:** Generated answers about reproductive health could cause harm if medically inaccurate. All outputs are grounded in retrieved training data reviewed by health professionals.
- **Language equity:** Low-resource African languages are underserved by NLP systems. This work contributes to closing that gap.
- **Cultural sensitivity:** Health topics around sexuality and reproduction require cultural awareness. The system respects local terminology and context.
- **Bias in evaluation:** ROUGE metrics favor lexical overlap and may disadvantage morphologically rich languages (e.g., Akan, Amharic).

## Technologies

- **Retrieval:** [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (278M params, 100+ languages)
- **Generation:** [Google Gemini 2.5 Flash](https://ai.google.dev/) via API
- **Fine-tuning:** [AfriTeVa V2](https://huggingface.co/castorini/afriteva_v2_base) (African-language T5), [mT5-base](https://huggingface.co/google/mt5-base)
- **Search:** FAISS (Facebook AI Similarity Search)
- **Evaluation:** ROUGE-score, custom whitespace tokenizer

## License

This project is submitted as coursework for Machine Learning Techniques I. The code is available for educational purposes.

## Acknowledgments

- [Zindi Africa](https://zindi.africa) for hosting the competition
- [AegisveriTas Project](https://aegisveritas.org) for inspiration
- Health professionals who curated the multilingual Q/A dataset