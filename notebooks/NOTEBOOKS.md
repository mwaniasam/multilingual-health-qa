# Notebooks Guide

## Main Pipeline (run in order)

| # | Notebook | Platform | What It Does | Score |
|---|----------|----------|-------------|-------|
| 01 | `01_eda_and_baseline.ipynb` | Kaggle | EDA + TF-IDF char n-gram retrieval baseline | 0.4908 |
| 02 | `02_mt5base_finetuning.ipynb` | Kaggle | mT5-base seq2seq fine-tuning (generation fails) | 0.2396 |
| 03 | `03_afriteva_finetuning.ipynb` | Kaggle | AfriTeVa V2 Africa-centric seq2seq | 0.2971 |
| 04 | `04_afrie5_retrieval_baseline.ipynb` | Colab | AfriE5-Large-instruct + MNRL + hard neg mining | 0.6545 |
| 05 | **`05_full_pipeline_v7.ipynb`** | Colab | **THE main notebook: bootstrap → CE reranker → V4/V6/V7** | **0.6908** |

### Notebook 05 is the core — it contains:
- **Cell 1:** Bootstrap (loads data, embeddings, indices, all helpers)
- **Cell 2:** Cross-encoder reranker v2 training (xlm-roberta-base, binary classification)
- **Cells 3-4:** CE-fed strategies (stitch + score-weighted MBR)
- **Cell 10:** V6 + V7 builder (QA-union CE-stitch on Amh_Eth)
- **Cells 13-15:** Qwen2.5-7B model reload + generation
- **Cell 16:** V7 → V8 comparison

To reproduce V7:
1. Run bootstrap cell (loads cached embeddings from Drive)
2. Load pre-trained models (AfriE5 + CE + Qwen from Drive)
3. Run V7 builder → `submission_v7.csv`

---

## Experiment Notebooks (in `experiments/`)

| # | Notebook | What It Tests | Verdict |
|---|----------|--------------|---------|
| 06 | `06_bm25_afrie5_fusion.ipynb` | BM25 + AfriE5 retrieval fusion | Rejected (0.6531) |
| 07 | `07_qwen_rag_reader.ipynb` | Qwen2.5-7B RAG reader fine-tuning | Adopted (Eng_Gha only) |
| 08 | `08_rouge_aligned_finetuning.ipynb` | ROUGE-aligned retriever fine-tuning | Rejected (catastrophic forgetting) |
| 09 | `09_retriever_finetuning_ft2.ipynb` | Global retriever fine-tune (FT2) | Adopted (via interpolation only) |
| 10 | `10_qwen_lora_training.ipynb` | Qwen LoRA training + V4 build | Adopted (Eng_Gha) |
| 11 | `11_complete_pipeline_v4.ipynb` | Complete V4 pipeline assembly | Superseded by V7 |
| 12 | `12_qwen_epoch2_3_probe.ipynb` | Qwen epochs 2-3 + strong lang test | Rejected (no improvement) |
| 13 | `13_perlang_adapters_probes.ipynb` | Per-language adapters + LB probes | Lug_Uga adopted, Eng_Uga rejected |
| 14 | `14_final_proposals_pruning.ipynb` | Cross-English retrieval, pruning | All rejected |

---

## How to run from scratch

```bash
# 1. Install dependencies
pip install -q -U torchao pylcs bitsandbytes transformers accelerate sentence-transformers faiss-cpu rouge-score peft trl

# 2. Mount Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# 3. Run notebook 05 bootstrap cell
# 4. Load models (AfriE5 + CE + Qwen)
# 5. Run V7 builder
```

Pre-trained models and cached embeddings are on Google Drive at:
`/content/drive/MyDrive/multilingual-health-qa/outputs/`
