# Artifacts Manifest — what to pull from Drive/Colab/Downloads

I cannot reach your Google Drive, Colab, or Downloads from the chat session. This is the checklist of everything that lives there, with a suggested destination in the local repo. Pull these down, then point Antigravity at the repo.

**Drive base:** `/content/drive/MyDrive/multilingual-health-qa`
**Outputs:** `BASE/outputs` · **Cache:** `BASE/outputs/mbr_cache`

## A. Submission files (from BASE/outputs) → `repo/submissions/`
- `submission_v7.csv` ← **BEST (0.6908), selected final**
- `submission_v4_final.csv` ← **selected final (hedge)**
- `submission_v6.csv`, `submission_v3.csv`, `submission_v2_compliant.csv` (history)
- any earlier `submission_*.csv` you want for the record

## B. Trained models (from BASE/outputs) → `repo/models/` (or keep on Drive; these are large)
- `afrie5-final-model/` ← retrieval backbone (fine-tuned AfriE5-Large)
- `ce-reranker-v2/` ← **THE cross-encoder** (xlm-roberta-base, used by V6/V7)
- `qwen-ft-health/` ← Qwen2.5-7B LoRA adapter, epoch 1 (Eng_Gha generation)
- `afrie5-lug_uga/` ← Lug_Uga per-language retriever adapter
- `afrie5-ft2/` ← global retriever fine-tune (used only via interpolation)
- (optional / rejected, keep for completeness) `qwen-ft-health-e3/`, `ce-reranker-large/`, `ce-dup-strong/`, `ce-reranker-v2-pl/` (if the pseudo-label cell ran)

## C. Caches (from BASE/outputs/mbr_cache) → `repo/cache/` (large .npy — Drive is fine as the source of truth)
- `emb_corpus.npy`, `emb_val.npy`, `emb_test.npy` ← AfriE5 Q→Q embeddings
- `emb_corpus_answers.npy` ← AfriE5 Q→A answer embeddings
- `ft2_corpus.npy`, `ft2_val.npy`, `ft2_test.npy` ← FT2 embeddings (for interpolation)
- `bge_corpus.npy`, `bge_val.npy`, `bge_test.npy` ← BGE-M3 leg
- `pl_Lug_Uga_corpus.npy`, `pl_Lug_Uga_val.npy`, `pl_Lug_Uga_test.npy` (+ `_idx.json`) ← Lug adapter
- `qa_emb_amh.npy` ← Amharic Q→(Q+A) leg (V7)
- `val_cands.pkl` / `val_cands_all`, `val_prep.pkl`, `val_union4.pkl` ← candidate pools + utilities
- `gem_emb_*` (non-compliant — keep only for the record, not for the final pipeline)

## D. Tuned-decision JSON + data (from BASE/outputs) → `repo/state/`
- `v3_strategy.json` ← per-language strategy table
- `interp_gate.json`, `interp_gate2.json` ← interpolation betas
- `aka_gen_gate.json`, `ft_langs.json` ← generation gates
- `ft_train_data.json` ← Qwen LoRA training text
- any `*_gate.json` / probe JSONs you have

## E. Raw competition data → `repo/data/`
- `Train.csv` (36,501 Q-A pairs), `Test.csv` (2,618), `SampleSubmission.csv`
- your val split definition (the even/odd holdout indices are derived deterministically from val order — document how `val_df` was split from Train)

## F. The Colab notebook itself → `repo/notebooks/`
- Export your working Colab `.ipynb`. **Its first big cell is the authoritative bootstrap** — the reference in `src/bootstrap_contract.py` is reconstructed from usage and must be reconciled against this real one.

## G. Deliverables still to produce (for the academic report / top-10 review)
- Single reproducible notebook: bootstrap → training (marked "ran once, artifacts on Drive") → V7 builder, seeded
- 7–10 min demo video
- Academic report (FINDINGS_AND_HANDOFF.md is the skeleton)
- ≥10 documented experiments (experiment_log.csv covers this)

## One security task
- **Rotate the API key exposed earlier** before any of this goes to a public repo or video.
