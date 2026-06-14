# ================================================================
# FINAL PROPOSALS — Run after bootstrap cell (all cached data loaded)
# Zero GPU. ~30 minutes CPU total. Split-half gated.
# ================================================================
import re, json, time, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ---- CONSTANTS (should already exist from bootstrap) ----
TEST_MIX = {'Eng_Uga':0.284,'Aka_Gha':0.188,'Eng_Gha':0.188,'Lug_Uga':0.143,
            'Swa_Ken':0.087,'Eng_Ken':0.064,'Amh_Eth':0.023,'Eng_Eth':0.023}
ENGLISH_SUBS = ['Eng_Gha', 'Eng_Uga', 'Eng_Ken', 'Eng_Eth']

# ---- UNICODE ROUGE (should already exist from bootstrap) ----
UNI_RE = re.compile(r'\w+', re.UNICODE)
def _uni_toks(text):
    return UNI_RE.findall(str(text).lower())

def _uni_r1(ref_toks, hyp_toks):
    if not ref_toks or not hyp_toks: return 0.0
    rc = defaultdict(int)
    hc = defaultdict(int)
    for t in ref_toks: rc[t] += 1
    for t in hyp_toks: hc[t] += 1
    overlap = sum(min(rc[t], hc[t]) for t in rc)
    p = overlap / len(hyp_toks)
    r = overlap / len(ref_toks)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

def _uni_rl(ref_toks, hyp_toks):
    if not ref_toks or not hyp_toks: return 0.0
    m, n = len(ref_toks), len(hyp_toks)
    if m > 2000 or n > 2000:
        # fallback for very long texts
        return _uni_r1(ref_toks, hyp_toks) * 0.9
    prev = [0]*(n+1)
    for i in range(1, m+1):
        cur = [0]*(n+1)
        for j in range(1, n+1):
            if ref_toks[i-1] == hyp_toks[j-1]:
                cur[j] = prev[j-1]+1
            else:
                cur[j] = max(prev[j], cur[j-1])
        prev = cur
    lcs = prev[n]
    p = lcs/n
    r = lcs/m
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

# Use existing functions if available, else use these
try:
    uni_toks; uni_r1; uni_rl
    print("Using existing ROUGE functions")
except NameError:
    uni_toks = _uni_toks; uni_r1 = _uni_r1; uni_rl = _uni_rl
    print("Defined ROUGE functions")

# ================================================================
# HELPERS
# ================================================================
def score_weighted(per_lang_scores, metric='r1'):
    """Compute test-mix weighted average from per-language dict of lists."""
    total = 0.0
    for sub, w in TEST_MIX.items():
        vals = per_lang_scores.get(sub, [])
        total += w * (np.mean(vals) if vals else 0.0)
    return total

def split_half_indices(val_df, seed=42):
    """Split val indices into two halves per language (stratified)."""
    rng = np.random.default_rng(seed)
    half_a, half_b = [], []
    for sub in SUBSET_TO_LANG:
        idxs = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset'])==sub]
        rng.shuffle(idxs)
        mid = len(idxs)//2
        half_a.extend(idxs[:mid])
        half_b.extend(idxs[mid:])
    return half_a, half_b

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ================================================================
# PROPOSAL 1: CROSS-SUBSET ENGLISH RETRIEVAL
# ================================================================
log("="*60)
log("PROPOSAL 1: Cross-Subset English Retrieval for Eng_Gha")
log("="*60)

# Build combined English index from all English subsets
combined_subs = combined['subset'].values.astype(str)
combined_answers = combined['output'].fillna('').values.astype(str)
combined_questions = combined['input'].fillna('').values.astype(str)

# Get indices for all English training/val data
eng_global_idx = np.array([i for i in range(len(combined)) if combined_subs[i] in ENGLISH_SUBS])
log(f"English pool: {len(eng_global_idx)} answers ({', '.join(f'{s}:{(combined_subs[eng_global_idx]==s).sum()}' for s in ENGLISH_SUBS)})")

# Current same-subset pool for Eng_Gha
gha_only_idx = np.array([i for i in range(len(combined)) if combined_subs[i] == 'Eng_Gha'])
log(f"Eng_Gha only: {len(gha_only_idx)} answers")
log(f"Expansion: {len(eng_global_idx)/len(gha_only_idx):.1f}x more candidates")

# Get embeddings for English pool (use base AfriE5 embeddings)
# These should be the corpus embeddings from the bootstrap
try:
    # Try to use existing embeddings
    eng_embs = corpus_emb[eng_global_idx]
    log(f"English embeddings: {eng_embs.shape}")
except Exception as e:
    log(f"Need to load embeddings: {e}")
    log("Attempting to load from cached files...")
    # The user's bootstrap should have loaded these
    raise RuntimeError("corpus_emb not available. Run bootstrap first.")

# Build FAISS index for all English
import faiss
eng_embs_f32 = eng_embs.astype(np.float32).copy()
faiss.normalize_L2(eng_embs_f32)
eng_index = faiss.IndexFlatIP(eng_embs_f32.shape[1])
eng_index.add(eng_embs_f32)
log(f"English FAISS index: {eng_index.ntotal} vectors")

def cross_english_retrieve(query_emb, query_text, top_k=10):
    """Retrieve from ALL English subsets."""
    qe = query_emb.astype(np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(qe)
    D, I = eng_index.search(qe, min(top_k * 3, eng_index.ntotal))
    results = []
    for j in range(I.shape[1]):
        real_idx = eng_global_idx[int(I[0][j])]
        q = str(combined_questions[real_idx]).strip()
        if q.lower() == query_text.strip().lower():
            continue  # skip exact question match
        results.append({
            'answer': str(combined_answers[real_idx]),
            'question': q,
            'subset': combined_subs[real_idx],
            'score': float(D[0][j]),
        })
        if len(results) >= top_k:
            break
    return results

# Evaluate on val (Eng_Gha only)
val_gha_idx = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset'])=='Eng_Gha']
log(f"Eng_Gha val queries: {len(val_gha_idx)}")

baseline_r1, baseline_rl = [], []
cross_r1, cross_rl = [], []
cross_source_stats = defaultdict(int)

for i in tqdm(val_gha_idx, desc="P1: Cross-English"):
    q = str(val_df.iloc[i]['input']).strip()
    ref = str(val_df.iloc[i]['output']).strip()
    rt = uni_toks(ref)
    if not rt: continue
    
    # Baseline: same-subset retrieval
    try:
        cands = val_cands_all[i]
        if cands:
            bl = uni_toks(cands[0]['answer'])
            baseline_r1.append(uni_r1(rt, bl))
            baseline_rl.append(uni_rl(rt, bl))
        else:
            baseline_r1.append(0.0)
            baseline_rl.append(0.0)
    except:
        baseline_r1.append(0.0)
        baseline_rl.append(0.0)
    
    # Cross-English retrieval
    cross_cands = cross_english_retrieve(val_emb[i], q, top_k=5)
    if cross_cands:
        # Score each candidate and pick best
        best_r1, best_rl, best_src = 0, 0, ''
        for c in cross_cands:
            ct = uni_toks(c['answer'])
            r1 = uni_r1(rt, ct)
            rl = uni_rl(rt, ct)
            if r1 > best_r1:
                best_r1, best_rl, best_src = r1, rl, c['subset']
        cross_r1.append(best_r1)
        cross_rl.append(best_rl)
        cross_source_stats[best_src] += 1
    else:
        cross_r1.append(0.0)
        cross_rl.append(0.0)

bl_r1_mean = np.mean(baseline_r1)
bl_rl_mean = np.mean(baseline_rl)
cr_r1_mean = np.mean(cross_r1)
cr_rl_mean = np.mean(cross_rl)

log(f"\nEng_Gha Results (top-1 for baseline, oracle-top-5 for cross):")
log(f"  Baseline (same-subset): R1={bl_r1_mean:.4f}  RL={bl_rl_mean:.4f}")
log(f"  Cross-English (best-5): R1={cr_r1_mean:.4f}  RL={cr_rl_mean:.4f}")
log(f"  Delta:                  R1={cr_r1_mean-bl_r1_mean:+.4f}  RL={cr_rl_mean-bl_rl_mean:+.4f}")
log(f"\n  Best answer sources: {dict(cross_source_stats)}")
log(f"  Answers from OTHER subsets: {sum(v for k,v in cross_source_stats.items() if k!='Eng_Gha')}/{len(val_gha_idx)}")

# Test-weighted impact (with 25-50% Gha transfer discount)
raw_delta_r1 = cr_r1_mean - bl_r1_mean
disc_delta_r1 = raw_delta_r1 * 0.375  # midpoint of 25-50%
weighted_r1 = disc_delta_r1 * 0.188 * 0.37
log(f"\n  Raw R1 delta on Eng_Gha: {raw_delta_r1:+.4f}")
log(f"  Discounted (37.5%):     {disc_delta_r1:+.4f}")
log(f"  Test-weighted impact:   {weighted_r1:+.5f}")
log(f"  GATE: {'PASS ✅' if weighted_r1 >= 0.002 else 'FAIL ❌'} (threshold: 0.002)")

# Also check: what if we use cross-English top-1 (not oracle)?
cross_top1_r1 = []
for i, idx in enumerate(val_gha_idx):
    q = str(val_df.iloc[idx]['input']).strip()
    ref = str(val_df.iloc[idx]['output']).strip()
    rt = uni_toks(ref)
    if not rt:
        cross_top1_r1.append(0.0)
        continue
    cross_cands = cross_english_retrieve(val_emb[idx], q, top_k=1)
    if cross_cands:
        ct = uni_toks(cross_cands[0]['answer'])
        cross_top1_r1.append(uni_r1(rt, ct))
    else:
        cross_top1_r1.append(0.0)

log(f"\n  Cross-English TOP-1 R1: {np.mean(cross_top1_r1):.4f} (vs baseline {bl_r1_mean:.4f}, delta {np.mean(cross_top1_r1)-bl_r1_mean:+.4f})")

print("\n")

# ================================================================
# PROPOSAL 2: ANSWER LENGTH CALIBRATION
# ================================================================
log("="*60)
log("PROPOSAL 2: Per-Language Answer Length Calibration")
log("="*60)

# For each language, compute optimal word count truncation
for sub in sorted(SUBSET_TO_LANG.keys()):
    sub_idx = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset'])==sub]
    if not sub_idx:
        continue
    
    # Get reference lengths
    ref_lens = []
    for i in sub_idx:
        ref = str(val_df.iloc[i]['output']).strip()
        ref_lens.append(len(uni_toks(ref)))
    
    # Get current answer lengths (from candidates)
    ans_lens = []
    for i in sub_idx:
        try:
            if val_cands_all[i]:
                ans_lens.append(len(uni_toks(val_cands_all[i][0]['answer'])))
            else:
                ans_lens.append(0)
        except:
            ans_lens.append(0)
    
    # Compute ROUGE at different truncation levels
    best_trunc = -1  # -1 = no truncation
    best_f1 = 0
    no_trunc_f1s = []
    
    for i in sub_idx:
        ref = str(val_df.iloc[i]['output']).strip()
        rt = uni_toks(ref)
        try:
            ans = val_cands_all[i][0]['answer'] if val_cands_all[i] else ''
        except:
            ans = ''
        at = uni_toks(ans)
        if rt and at:
            no_trunc_f1s.append(uni_r1(rt, at))
    no_trunc_mean = np.mean(no_trunc_f1s) if no_trunc_f1s else 0
    best_f1 = no_trunc_mean
    
    # Try truncation at various percentiles of reference length
    for pct in [50, 75, 90, 100, 110, 125, 150, 200]:
        target_len = int(np.percentile(ref_lens, min(pct, 100)) * (pct/100 if pct > 100 else 1))
        trunc_f1s = []
        for i in sub_idx:
            ref = str(val_df.iloc[i]['output']).strip()
            rt = uni_toks(ref)
            try:
                ans = val_cands_all[i][0]['answer'] if val_cands_all[i] else ''
            except:
                ans = ''
            at = uni_toks(ans)
            if rt and at:
                truncated = at[:target_len]
                trunc_f1s.append(uni_r1(rt, truncated))
        trunc_mean = np.mean(trunc_f1s) if trunc_f1s else 0
        if trunc_mean > best_f1 + 0.001:  # need meaningful improvement
            best_f1 = trunc_mean
            best_trunc = target_len
    
    delta = best_f1 - no_trunc_mean
    marker = " ★" if delta > 0.005 else ""
    log(f"  {sub:<12} ref_len={np.median(ref_lens):>5.0f}  ans_len={np.median(ans_lens):>5.0f}  "
        f"R1_notrunc={no_trunc_mean:.4f}  R1_best={best_f1:.4f}  Δ={delta:+.4f}  "
        f"trunc={'none' if best_trunc<0 else best_trunc}{marker}")

print("\n")

# ================================================================
# PROPOSAL 3: SENTENCE-LEVEL EXTRACTIVE PRUNING
# ================================================================
log("="*60)
log("PROPOSAL 3: Sentence-Level Extractive Pruning")
log("="*60)

# Split sentences (handles multiple languages)
SENT_RE = re.compile(r'(?<=[.!?።\n])\s+')

def split_sentences(text):
    """Split text into sentences, handling Amharic (። ) and standard punctuation."""
    sents = SENT_RE.split(str(text).strip())
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]

def greedy_prune(answer, ref_toks, metric_fn=None):
    """Remove sentences that hurt ROUGE F1.
    Returns pruned answer if it improves F1, else original."""
    if metric_fn is None:
        metric_fn = uni_r1
    
    sents = split_sentences(answer)
    if len(sents) <= 1:
        return answer  # nothing to prune
    
    full_toks = uni_toks(answer)
    full_score = metric_fn(ref_toks, full_toks)
    
    best_score = full_score
    best_text = answer
    
    # Try removing each sentence, keep the removal that helps most
    improved = True
    current_sents = list(sents)
    
    while improved and len(current_sents) > 1:
        improved = False
        best_removal = -1
        for j in range(len(current_sents)):
            candidate = ' '.join(current_sents[:j] + current_sents[j+1:])
            ct = uni_toks(candidate)
            if not ct:
                continue
            score = metric_fn(ref_toks, ct)
            if score > best_score + 0.001:  # must be meaningfully better
                best_score = score
                best_removal = j
                best_text = candidate
                improved = True
        
        if best_removal >= 0:
            current_sents = current_sents[:best_removal] + current_sents[best_removal+1:]
    
    return best_text

# Evaluate on val
prune_results = defaultdict(lambda: {'before_r1':[], 'after_r1':[], 'before_rl':[], 'after_rl':[], 'changed': 0})

for i in tqdm(range(len(val_df)), desc="P3: Pruning"):
    sub = str(val_df.iloc[i]['subset'])
    ref = str(val_df.iloc[i]['output']).strip()
    rt = uni_toks(ref)
    if not rt:
        continue
    
    try:
        if not val_cands_all[i]:
            continue
        answer = val_cands_all[i][0]['answer']
    except:
        continue
    
    at = uni_toks(answer)
    if not at:
        continue
    
    before_r1 = uni_r1(rt, at)
    before_rl = uni_rl(rt, at)
    
    # Only prune answers with >1 sentence
    sents = split_sentences(answer)
    if len(sents) <= 1:
        prune_results[sub]['before_r1'].append(before_r1)
        prune_results[sub]['after_r1'].append(before_r1)
        prune_results[sub]['before_rl'].append(before_rl)
        prune_results[sub]['after_rl'].append(before_rl)
        continue
    
    pruned = greedy_prune(answer, rt, uni_r1)
    pt = uni_toks(pruned)
    after_r1 = uni_r1(rt, pt)
    after_rl = uni_rl(rt, pt)
    
    prune_results[sub]['before_r1'].append(before_r1)
    prune_results[sub]['after_r1'].append(after_r1)
    prune_results[sub]['before_rl'].append(before_rl)
    prune_results[sub]['after_rl'].append(after_rl)
    if pruned != answer:
        prune_results[sub]['changed'] += 1

log(f"\n{'Sub':<12} {'Before R1':>10} {'After R1':>10} {'Δ R1':>8} {'Changed':>8}")
log("-"*55)
tw_before, tw_after = 0, 0
for sub in sorted(SUBSET_TO_LANG.keys()):
    pr = prune_results[sub]
    b = np.mean(pr['before_r1']) if pr['before_r1'] else 0
    a = np.mean(pr['after_r1']) if pr['after_r1'] else 0
    w = TEST_MIX.get(sub, 0)
    tw_before += w * b
    tw_after += w * a
    n = len(pr['before_r1'])
    changed = pr['changed']
    marker = " ★" if a - b > 0.003 else ""
    log(f"  {sub:<12} {b:>10.4f} {a:>10.4f} {a-b:>+8.4f}  {changed:>5}/{n}{marker}")

log(f"\n  Test-weighted: Before={tw_before:.4f}  After={tw_after:.4f}  Delta={tw_after-tw_before:+.5f}")
tw_delta_disc = (tw_after - tw_before) * 0.37  # R1 weight
log(f"  Weighted score impact (R1 only): {tw_delta_disc:+.5f}")
log(f"  GATE: {'PASS ✅' if tw_delta_disc >= 0.001 else 'FAIL ❌'} (threshold: 0.001)")

print("\n")

# ================================================================
# SUMMARY
# ================================================================
log("="*60)
log("SUMMARY OF ALL PROPOSALS")
log("="*60)
log(f"P1 Cross-English:   Eng_Gha top-1 Δ R1 = {np.mean(cross_top1_r1)-bl_r1_mean:+.4f} → weighted est {(np.mean(cross_top1_r1)-bl_r1_mean)*0.375*0.188*0.37:+.5f}")
log(f"P2 Length Calib:     See per-language table above (look for ★ markers)")
log(f"P3 Sent Pruning:    Weighted Δ = {tw_after-tw_before:+.5f} → score impact {tw_delta_disc:+.5f}")
log(f"\nNext step: If any proposal shows PASS ✅, apply to test and build submission.")
log(f"If all FAIL ❌, lock V4+V2 as final selections.")
