"""
bootstrap_contract.py
=====================
The pipeline runs on a single "bootstrap" cell (in your Colab notebook) that loads
data, builds embeddings/indices, and defines a set of helper functions. That cell
lives on your Drive and is the AUTHORITATIVE source. This file documents the CONTRACT
(what the bootstrap must provide) and gives verified/faithful implementations of the
helpers so Antigravity can reconcile and run locally.

>>> RECONCILE every function below against your real Colab bootstrap before trusting it.
    uni_toks / uni_r1 / uni_rl / mbr_idx are standard and high-confidence.
    uni_prep / uni_stitch are faithful reconstructions from observed behavior — VERIFY.
"""

import re
import numpy as np
from collections import Counter
try:
    import pylcs  # used for ROUGE-L LCS length
except ImportError:
    pylcs = None

# ============================================================================
# CONSTANTS (verified)
# ============================================================================
CAP = 400                                   # token cap applied as uni_toks(text)[:CAP]
_UNI = re.compile(r"\w+", re.UNICODE)         # the tokenizer that matches the fixed scorer
SUBSET_TO_LANG = {
    "Aka_Gha": "Akan (Ghana)",   "Amh_Eth": "Amharic (Ethiopia)",
    "Eng_Eth": "English (Ethiopia)", "Eng_Gha": "English (Ghana)",
    "Eng_Ken": "English (Kenya)", "Eng_Uga": "English (Uganda)",
    "Lug_Uga": "Luganda (Uganda)", "Swa_Ken": "Kiswahili (Kenya)",
}
SUB_COLS = ["ID", "TargetR1F1", "TargetRLF1", "TargetLLM"]
K_CANDIDATES = 15

# ============================================================================
# SCORING HELPERS (high-confidence; standard ROUGE under a unicode tokenizer)
# ============================================================================
def uni_toks(text):
    """Unicode word tokenizer matching the organizers' fixed scorer."""
    return _UNI.findall(str(text).lower())

def uni_r1(ref_toks, cand_toks):
    """ROUGE-1 F1 on token multisets (ref_toks, cand_toks already capped)."""
    if not ref_toks or not cand_toks:
        return 0.0
    overlap = sum((Counter(ref_toks) & Counter(cand_toks)).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(cand_toks)
    r = overlap / len(ref_toks)
    return 2 * p * r / (p + r)

def uni_rl(ref_toks, cand_toks):
    """ROUGE-L F1 via LCS length (pylcs)."""
    if not ref_toks or not cand_toks:
        return 0.0
    if pylcs is not None:
        lcs = pylcs.lcs_sequence_length(" ".join(ref_toks), " ".join(cand_toks))
        # pylcs operates on characters above; for token-LCS use the index variant:
        lcs = _token_lcs(ref_toks, cand_toks)
    else:
        lcs = _token_lcs(ref_toks, cand_toks)
    if lcs == 0:
        return 0.0
    p = lcs / len(cand_toks)
    r = lcs / len(ref_toks)
    return 2 * p * r / (p + r)

def _token_lcs(a, b):
    """Token-level LCS length (DP). Replace with pylcs index API if you cached that."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]

# ============================================================================
# MBR SELECTION (faithful reconstruction — verify signature against Colab)
# ============================================================================
def uni_prep(pool):
    """
    Precompute MBR machinery for a candidate pool.
    pool: list of {'answer': str, 'sim': float, 'idx': int} (top-1 first).
    Returns (dd, ddw, u1, uL):
      dd  = list of answer strings
      ddw = consensus weights (softmaxed retrieval sims)
      u1  = per-candidate ROUGE-1 consensus utility vector
      uL  = per-candidate ROUGE-L consensus utility vector
    """
    dd = [c["answer"] for c in pool]
    sims = np.array([c.get("sim", 0.0) for c in pool], dtype=np.float64)
    w = np.exp(sims * 5.0)             # sharpen toward retrieval confidence
    ddw = w / (w.sum() + 1e-9)
    toks = [uni_toks(a)[:CAP] for a in dd]
    n = len(dd)
    u1 = np.zeros(n); uL = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u1[i] += ddw[j] * uni_r1(toks[j], toks[i])
            uL[i] += ddw[j] * uni_rl(toks[j], toks[i])
    return dd, ddw, u1, uL

def mbr_idx(util, ddw, alpha, margin):
    """Guarded MBR pick: argmax(util + alpha*ddw), but keep top-1 unless margin exceeded."""
    scored = util + alpha * ddw
    best = int(np.argmax(scored))
    return best if (scored[best] - scored[0]) > margin else 0

# ============================================================================
# EXTRACTIVE STITCHER (faithful reconstruction — VERIFY against Colab)
# ============================================================================
def uni_stitch(pool, lam, sub):
    """
    Greedy extractive stitcher: assemble verbatim sentences from the pool to maximize
    expected ROUGE-1 F1 against the weighted candidate-consensus token distribution,
    with a per-language length prior controlled by lam.
    NOTE: this is a behavior-faithful reconstruction. Your Colab original is authoritative;
    in particular the consensus token target, the per-language ref-length prior, and the
    sentence splitter must match. Stitch output feeds the R1/identical column only.
    """
    import itertools
    dd, ddw, _, _ = uni_prep(pool)
    # weighted consensus unigram distribution (the extraction target)
    target = Counter()
    for a, w in zip(dd, ddw):
        for t in set(uni_toks(a)[:CAP]):
            target[t] += w
    # candidate sentences (verbatim, deduped)
    sents = []
    for a in dd:
        for s in re.split(r"(?<=[.!?])\s+", str(a).strip()):
            s = s.strip()
            if s and s not in sents:
                sents.append(s)
    if not sents:
        return dd[0]
    # per-language target length prior (tune lam externally; uni_stitch_gate carries it)
    ref_len = max(1, int(round(lam * np.mean([len(uni_toks(a)[:CAP]) for a in dd]))))
    chosen, chosen_toks = [], []
    remaining = list(sents)
    while remaining and len(chosen_toks) < ref_len:
        def gain(s):
            t = uni_toks(s)[:CAP]
            cand = chosen_toks + t
            ov = sum((Counter(cand) & target).values())
            p = ov / max(1, len(cand))
            r = ov / max(1, sum(target.values()))
            return 2 * p * r / (p + r + 1e-9)
        best = max(remaining, key=gain)
        if not best:
            break
        chosen.append(best); chosen_toks += uni_toks(best)[:CAP]
        remaining.remove(best)
    return " ".join(chosen) if chosen else dd[0]

# ============================================================================
# CONTRACT — objects the real bootstrap MUST provide (loaded/built in Colab)
# ============================================================================
"""
DATA
  train_df, val_df, test_df          : pandas DataFrames (Train split into train/val)
  combined                           : concatenated corpus rows used for retrieval
  questions_raw, answers_raw         : lists aligned to `combined`
  subsets_raw                        : list of language-subset tags aligned to `combined`
  corpus_q_stripped                  : [q.strip() for q in questions_raw]
  val_qs, test_qs                    : question strings for val/test
  test_subs                          : language-subset tag per test row

EMBEDDINGS / INDICES
  corpus_emb, val_emb, test_emb      : AfriE5 Q->Q embeddings (np.float32, normalized)
  lang_indices[sub] -> (faiss_index, mask_list)   # per-language FAISS over corpus_emb
  build_lang_idx(emb) -> dict        # helper to build per-language indices for a new leg

RETRIEVAL
  get_same_lang_candidates(q_text, q_emb, subset, k=K_CANDIDATES, exclude_exact=True)
      -> list[{'answer','sim','idx'}]   # same-language top-k; set exclude_exact=False at TEST
  union4(q_text, afri_q, gem_q, bge_q, subset, exclude_exact=True)
      -> RRF union of 4 legs; 'sim' carries AfriE5 cosine (NOT RRF) for calibrated MBR

TUNED DECISIONS (hardcoded in bootstrap; reproducible)
  choice[sub] -> (tag, alpha, margin)              # 'tag' in {'2leg','4leg'}
  uni_stitch_gate[sub] -> {'use':bool, 'lam':float, 'pool':'2leg'|'4leg'}

CACHES (np.load from mbr_cache; see ARTIFACTS_MANIFEST.md)
  val_cands_all, val_prep, v4c (4-leg val pools), val_refscores, P (per-tag prep/ref)
"""
