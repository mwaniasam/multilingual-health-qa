# Multilingual Health QA Challenge — Complete Findings & Handoff

**Competitor:** mwaniasam (Samuel Mwania)
**Competition:** Zindi — Multilingual Health Question Answering in Low-Resource African Languages (ITU / HASH / Makerere)
**Final public best:** **0.6908** (V7) · public rank ~13–15 · climbed from 0.6545
**Leader (Brainiac):** 0.7285 · 2nd–3rd (charmq, Preferred Networks engineer / Kaggle Master): ~0.713
**Deadline:** 22 June 2026 · **Prizes:** $2,500 / $1,500 / $1,000
**Status at handoff:** competition work concluded; pipeline audited to its floor; this document is the authoritative record for merging into the local repo.

> This document consolidates everything from the Claude coaching sessions. It does **not** include the prior local Antigravity pipeline (not visible to the session) — merging the two is the local task. See `README.md` and `ARTIFACTS_MANIFEST.md`.

---

## 1. Ground truth about the competition (verified, not assumed)

| Fact | Detail | Consequence |
|---|---|---|
| **Metric weights** | Score = 0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·LLM-Judge. Verified: reproduces our LB and Brainiac's to 4 decimals. | ROUGE is 74% of the score. Judge work caps at 0.26 leverage. |
| **Identical-columns rule** | Info page: the 3 target columns "should be identical." Scorer does NOT enforce it (columns scored independently), but top-10 get code review. | Final selected files use identical columns. Identical-column files also *scored best* on the judge anyway. |
| **Open-source-only rule** | "Solution must use publicly-available, open-source packages only… no paid services." | Gemini (generation + embeddings) is OUT of the final solution. Used only for early experimentation. |
| **Tokenizer fix (21 May)** | Organizers fixed ROUGE tokenization for Ge'ez (Amharic) and Akan ɛ/ɔ. LB rescored before we joined. | Local `rouge_score` still broken for these scripts. All local scoring uses a Unicode tokenizer: `re.compile(r'\w+', re.UNICODE)`, CAP=400. Amharic true baseline 0.21 (not 0.04); Akan 0.33 (not 0.39). |
| **Hidden 4th metric (phase 2)** | Top solutions also evaluated with AfroLM BertScore F1 (semantic similarity). | Favors semantically faithful answers; avoid degenerate ROUGE-gaming. |
| **Judge** | LLM-as-judge, reference-anchored (paraphrasing away from the reference costs points). | Identical-column consensus answers score 0.80–0.82, *higher* than top-1 or Gemini refinement. Judge noise ±0.005. |
| **Test language mix** | Eng_Uga 28.4%, Aka_Gha 18.8%, Eng_Gha 18.8%, Lug_Uga 14.3%, Swa_Ken 8.7%, Eng_Ken 6.4%, Amh_Eth 2.3%, Eng_Eth 2.3%. | 80% of test is 4 languages. Amharic/Eng_Eth effort is nearly worthless by weight. Val mix differs → all val numbers must be test-mix reweighted. |
| **Public/private** | Public LB = 30% (~785 q); private = 70% decides rank. Two selected submissions count. | Per-language public deltas are noise; val (test-weighted) is the optimization target. |
| **Submission budget** | 5/day, 50 total. ~39 used. | Daily cap was the binding constraint, not the 50 total. |

---

## 2. Score progression (public LB)

| # | File | Score | R1 | RL | LLM | What changed |
|---|---|---|---|---|---|---|
| 0 | AfriE5 Q→Q top-1 (pre-engagement) | 0.6545 | 0.627 | 0.561 | 0.775 | baseline |
| 1 | submission_mbr | 0.6597 | 0.6502 | 0.5848 | 0.78 | MBR selection + same-lang + exact-duplicate recovery |
| 2 | submission_mbr_gemini | 0.6623 | 0.6502 | 0.5848 | 0.7899 | Gemini grounded refinement (later dropped — non-compliant) |
| 3 | submission_stitch_gated | 0.6650 | 0.6607 | 0.5848 | 0.7854 | Extractive stitcher (Aka/Eng_Gha R1) |
| 4 | submission_union | 0.6652 | 0.6628 | 0.5857 | 0.7819 | 4-leg pool union (Eng_Gha) |
| 5 | submission_uni_split | 0.6670 | 0.6632 | 0.5875 | 0.7854 | Unicode tokenizer rebuild; Amharic/Gha stitch |
| 6 | submission_v2_compliant | 0.6843 | 0.6751 | 0.6085 | 0.8052 | **Identical columns + Eng_Gha fine-tuned generation.** Biggest single jump. |
| 7 | submission_v3 | 0.6850 | 0.6781 | 0.6065 | 0.8065 | Per-language strategy table |
| 8 | **submission_v4_final** | **0.6898** | 0.6791 | 0.6077 | 0.8218 | + embedding interpolation (Eng_Uga/Lug_Uga/Swa_Ken) + Aka re-decision |
| 9 | submission_v6 | ~0.689 | — | — | — | + CE-base-stitch (Aka/Amh) + Lug_Uga adapter + Swa β |
| 10 | **submission_v7** | **0.6908 (BEST)** | 0.6813 | 0.6097 | 0.8199 | + Amh_Eth QA-union CE-stitch |
| 11 | submission_v8 | ~0.6894 (REJECTED) | — | — | — | V7 + QA-union generation swap on Eng_Gha: ROUGE-holdout +0.017 but judge column −0.0063 → net −0.0014 on LB |

> Numbers for files 9 and 11 are approximate; `docs/experiment_log.csv` and your Zindi submissions page are authoritative. V4 (0.6898) and V7 (0.6908) are the verified anchors and the two selected finals.

**Final selections:** **V7** (top line) + **V4** (composition hedge — higher judge column 0.8218, different Eng_Gha treatment, for private-split diversity).

---

## 3. The winning architecture (what V7 actually is)

V7 = a retrieval-and-selection pipeline, fully open-source, identical columns. Per-language routing, every component holdout-gated:

1. **Retrieval backbone:** AfriE5-Large-instruct (McGill-NLP), fine-tuned Q→Q with MNRL + hard negatives. Confirmed SOTA African embedder (AfriMTEB paper). This is the core; the 0.6545 baseline was already this model at top-1.
2. **MBR consensus selection** with per-language guarded override (τ tuned; strong languages correctly lock to top-1). ROUGE-1 utility for the R1 column, ROUGE-L for the RL column; combined 0.5·R1+0.5·RL for the identical-column answer.
3. **Exact-duplicate recovery** at test time (`exclude_exact=False`) — free points (~+0.01).
4. **Greedy extractive stitcher** — sentence-level, verbatim only, maximizing expected ROUGE-1 F1 vs the weighted candidate-consensus token distribution, with a per-language length prior λ. Wins on multi-valid-answer languages; gated off elsewhere.
5. **Fine-tuned Qwen2.5-7B (LoRA r=16, epoch 1)** generation — deployed for **Eng_Gha only**, where generation *beats* the retrieval oracle. The single biggest discovery. (Epochs 2–3 rejected; generation on strong languages rejected — wall is task structure, not undertraining.)
6. **Cross-encoder reranker (`xlm-roberta-base`, binary classification)** → feeds the stitch on **Aka_Gha** and **Amh_Eth**. Re-orders candidate pools so the stitcher extracts from better-ranked answers.
7. **Per-language retriever adapter (Lug_Uga, answer-similarity supervision)**, deployed via embedding interpolation β·AfriE5 + (1−β)·adapter, β=0.8.
8. **Embedding interpolation** β·AfriE5 + (1−β)·FT2 for Eng_Uga / Lug_Uga / Swa_Ken (gated betas).
9. **QA-union CE-stitch** on Amh_Eth (a fifth Q→(Q+A) retrieval leg, unioned into the Amharic pool).

Everything else (the strong languages: Eng_Uga, Eng_Ken, Eng_Eth) routes to combined-MBR on the AfriE5 pool — top-1 is already near-oracle there.

---

## 4. Experiment ledger — every verdict

### Adopted (in V7)
MBR selection · exact-duplicate recovery · extractive stitcher · unicode-tokenizer rebuild · Qwen ft generation (Eng_Gha) · CE-base → stitch (Aka_Gha, Amh_Eth) · Lug_Uga per-language adapter (β=0.8) · embedding interpolation (Eng_Uga/Lug_Uga/Swa_Ken βs) · QA-union CE-stitch (Amh_Eth).

### Rejected / measured-closed (do not re-run as-is)
- **Cross-encoder ROUGE-regression** (week 1): wrong objective (predicting ROUGE vs unknown reference). The *classification* reformulation is what later worked.
- **CE-large (`xlm-roberta-large`)**: loss 0.503 → 0.513, never converged; lost to CE-base everywhere.
- **CE dup-group supervision (strong languages)**: every language picked max margin and still lost — dense top-1 already surfaces the duplicate; nothing to rerank.
- **BGE-reranker-v2-m3 (pretrained)**: did not beat the in-domain CE on its home turf.
- **Global retriever fine-tune (FT2)**: catastrophic forgetting on strong languages; salvaged only via interpolation.
- **Eng_Uga per-language adapter**: 942 triplets too thin vs a 0.87 top-1; β=0.0.
- **Generation on strong languages (Lug_Uga, Eng_Uga)**: Δ ≈ −0.089 at epochs 1 and 3 — must out-copy a 0.83 top-1; structural wall.
- **Qwen epochs 2–3**: only beneficiaries were already-banked Gha gains (transfer at 25%).
- **Gemini refinement / Gemini embeddings**: only +0.01 on the real judge, and non-compliant; dropped.
- **HyDE (answer-embedding retrieval from generations)**: dead on locked languages; below deployed strategies on Gha.
- **Judge-aware tie-break (longest-of-near-tied)**: −0.006 to −0.44; utility ties ≠ quality ties.
- **Full-length LCS rerank (unguarded)**: −0.21 to −0.70; duplicates outvote a strong top-1.
- **K-sweep, stitch-λ re-tune, interpolation round 2, cross-lingual Aka→Eng_Gha pool**: optima already found / no movement. (One stitch-λ re-tune surfaced a self-retrieval **leak bug** — caught and fixed; V7 is clean.)
- **mT5-base (0.240), AfriTeVa V2 (0.297), BM25+dense RRF (0.6531)**: week-1 dead ends.
- **V8 (QA-union generation swap on Eng_Gha)**: ROUGE up on holdout, judge column down on LB; net −0.0014. Rejected.
- **Pseudo-labeling CE (this session's last experiment)**: not completed — competition called. Cells preserved in `src/training_cells.py` (Section E) if ever revisited. Honest prior: bounded gain (CE feeds only 2 languages).

### Confirmed by external research (no hidden technique exists)
AfriE5-large-instruct = best-in-class African embedder. Winning recipes for analogous health-QA shared tasks (biomedical QA arXiv 2507.05577; AraHealthQA arXiv 2508.20047) = dense retrieval + cross-encoder rerank + LLM — i.e. exactly this stack. The one genuinely-untried open lever surfaced was **Qwen3-Embedding-8B** as an extra retrieval leg (Apache-2.0); priced ~30% odds, never run.

---

## 5. Calibration constants (empirical — keep these)

- Local sim (unicode, test-mix weighted) runs **~+0.005 optimistic** vs LB.
- Transfer rates: **selection ~1:1; stitch ~0.5:1; Gha-language gains 25–50%.**
- Judge noise **±0.005** between identical-content files.
- **Submission rule that worked 36×:** split-half holdout gate (tune on even val indices, confirm on odd), auto-revert, sim-delta threshold before spending a slot. One mechanism change per submission. **This discipline is the entire +0.036 climb.**

---

## 6. Per-language oracle table (val, original tokenizer — directional)

| Language | N | Current R1 | Oracle R1 | Gap | Note |
|---|---|---|---|---|---|
| Swa_Ken | 2588 | 0.946 | 0.983 | 0.037 | near ceiling |
| Eng_Ken | 2470 | 0.897 | 0.978 | 0.080 | near ceiling |
| Eng_Uga | 9312 | 0.862 | 0.950 | 0.088 | 28.4% of test; retrieval-locked |
| Lug_Uga | 4229 | 0.826 | 0.929 | 0.104 | adapter + interp adopted |
| Eng_Eth | 4479 | 0.689 | 0.785 | 0.096 | 2.3% of test |
| Aka_Gha | 5569 | 0.391 | 0.519 | 0.129 | CE-stitch adopted |
| Eng_Gha | 5547 | 0.336 | 0.447 | 0.111 | ft-generation adopted |
| Amh_Eth | 2307 | 0.042* | 0.110* | 0.068 | *broken tokenizer; true baseline 0.21 |

Probe decomposition of the public LB judge column: Uga ≈ 0.89, Ken ≈ 0.92, **Gha ≈ 0.70 (structural, strategy-insensitive)** — Gha's open-ended questions are the hard ceiling.

---

## 7. What a senior engineer / Antigravity should consider next (if ever resumed)

Honestly ranked, all gated, none likely > +0.005:
1. **Qwen3-Embedding-8B** as a 5th retrieval leg (the one untried open-source retriever; ~30% odds; 8B is tight on 8GB — run quantized or on Colab).
2. **Length feature inside the CE reranker** (AraHealthQA winner used length + question-similarity as rerank features).
3. Pseudo-labeling CE (cells ready; bounded EV).
4. Prefix-stripping on retrieved answers; Eng_Gha resample-then-rerank.

The frontier is otherwise measured-closed. The realistic remaining fight was always *best private-split position*, not #1 (0.0387 gap, arithmetically unreachable in the time left).

---

## 8. Operating discipline (non-negotiable — learned the hard way)

- Holdout-validate (split-half) every change; auto-revert on holdout failure.
- Test-mix reweight every val number; subtract +0.005 sim optimism; submit only at sim-delta ≥ threshold.
- One mechanism change per submission; never ship unvalidated ideas.
- Cache everything to Drive; hardcode tuned decisions in the bootstrap cell (restart-proof).
- Mine/fine-tune on **train rows only**; val is the judge (never leak).
- Watch for self-retrieval leaks: always filter `corpus_q_stripped[ci] == query` when building pools from any new leg.
- Top-10 private triggers a 72h code+report demand — everything in the final files must re-run from the bootstrap.
- **Rotate the API key that was exposed earlier before the repo/demo go public.**

---
*Generated for the local-repo handoff. Pair with `src/` (code), `docs/experiment_log.csv` (ledger), and `docs/ARTIFACTS_MANIFEST.md` (Drive artifacts to pull).*
