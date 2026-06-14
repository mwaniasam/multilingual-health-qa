# ================================================================
# PROPOSAL 3 → TEST IMPLEMENTATION
# Sentence pruning using CONSENSUS proxy (no reference needed)
# ================================================================
# Run AFTER bootstrap. Uses cached val_cands_all, test candidates, etc.

import re, json, time, numpy as np, pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
UNI_RE = re.compile(r'\w+', re.UNICODE)
SENT_RE = re.compile(r'(?<=[.!?።\n])\s+')

def uni_toks_local(text):
    return UNI_RE.findall(str(text).lower())

def uni_r1_local(ref_toks, hyp_toks):
    if not ref_toks or not hyp_toks: return 0.0
    rc = Counter(ref_toks); hc = Counter(hyp_toks)
    overlap = sum(min(rc[t], hc[t]) for t in rc)
    p = overlap / len(hyp_toks); r = overlap / len(ref_toks)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

# Use existing if available
try: uni_toks; uni_r1
except NameError: uni_toks = uni_toks_local; uni_r1 = uni_r1_local

def split_sents(text):
    sents = SENT_RE.split(str(text).strip())
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]

# ================================================================
# CONSENSUS-BASED PRUNING (reference-free)
# For each answer, remove sentences that have lowest overlap
# with OTHER top-K candidates' answers (the consensus)
# ================================================================
def build_pseudo_ref(candidates, exclude_idx=0):
    """Build a pseudo-reference from top-K candidates (excluding the selected one).
    Concatenate all other candidates to form a token bag."""
    tokens = []
    for j, c in enumerate(candidates):
        if j == exclude_idx:
            continue
        tokens.extend(uni_toks(c.get('answer', '')))
    return tokens

def consensus_prune(answer, candidates, min_sents=1):
    """Remove sentences from answer that have lowest overlap with other candidates.
    Reference-free: uses top-K consensus as proxy for reference."""
    sents = split_sents(answer)
    if len(sents) <= min_sents:
        return answer
    
    # Build pseudo-reference from other candidates
    pseudo_ref = build_pseudo_ref(candidates, exclude_idx=0)
    if not pseudo_ref:
        return answer
    
    # Also use query overlap as secondary signal
    full_toks = uni_toks(answer)
    full_score = uni_r1(pseudo_ref, full_toks)
    
    # Greedily remove sentences that improve consensus F1
    current_sents = list(sents)
    best_score = full_score
    best_text = answer
    
    improved = True
    while improved and len(current_sents) > min_sents:
        improved = False
        best_removal = -1
        for j in range(len(current_sents)):
            candidate_text = ' '.join(current_sents[:j] + current_sents[j+1:])
            ct = uni_toks(candidate_text)
            if not ct:
                continue
            score = uni_r1(pseudo_ref, ct)
            if score > best_score + 0.002:  # threshold to avoid noise
                best_score = score
                best_removal = j
                best_text = candidate_text
                improved = True
        if best_removal >= 0:
            current_sents = current_sents[:best_removal] + current_sents[best_removal+1:]
    
    return best_text

# ================================================================
# VALIDATE ON VAL: Consensus pruning vs Oracle pruning
# ================================================================
log("VALIDATING: Consensus-based pruning (reference-free) on val")
log("="*60)

TEST_MIX = {'Eng_Uga':0.284,'Aka_Gha':0.188,'Eng_Gha':0.188,'Lug_Uga':0.143,
            'Swa_Ken':0.087,'Eng_Ken':0.064,'Amh_Eth':0.023,'Eng_Eth':0.023}

cons_results = defaultdict(lambda: {'before_r1':[], 'after_r1':[], 'changed': 0})

for i in tqdm(range(len(val_df)), desc="Consensus prune val"):
    sub = str(val_df.iloc[i]['subset'])
    ref = str(val_df.iloc[i]['output']).strip()
    rt = uni_toks(ref)
    if not rt: continue
    
    try:
        cands = val_cands_all[i]
        if not cands: continue
        answer = cands[0]['answer']
    except:
        continue
    
    at = uni_toks(answer)
    if not at: continue
    
    before = uni_r1(rt, at)
    
    # Consensus prune (reference-free)
    pruned = consensus_prune(answer, cands)
    pt = uni_toks(pruned)
    after = uni_r1(rt, pt)
    
    cons_results[sub]['before_r1'].append(before)
    cons_results[sub]['after_r1'].append(after)
    if pruned != answer:
        cons_results[sub]['changed'] += 1

log(f"\n{'Sub':<12} {'Before':>8} {'After':>8} {'Δ R1':>8} {'Changed':>10}")
log("-"*55)
tw_before, tw_after = 0, 0
for sub in sorted(SUBSET_TO_LANG.keys()):
    cr = cons_results[sub]
    b = np.mean(cr['before_r1']) if cr['before_r1'] else 0
    a = np.mean(cr['after_r1']) if cr['after_r1'] else 0
    w = TEST_MIX.get(sub, 0)
    tw_before += w * b; tw_after += w * a
    n = len(cr['before_r1']); c = cr['changed']
    marker = " ★" if a - b > 0.003 else ""
    log(f"  {sub:<12} {b:>8.4f} {a:>8.4f} {a-b:>+8.4f} {c:>5}/{n}{marker}")

delta = tw_after - tw_before
score_impact = delta * 0.37
log(f"\n  Test-weighted R1: Before={tw_before:.4f}  After={tw_after:.4f}  Δ={delta:+.5f}")
log(f"  Score impact (R1 only): {score_impact:+.5f}")

# Apply Gha transfer discount
# Strong languages transfer ~1:1, Gha ~25-50%
# But pruning helps ALL languages, so blended discount is lighter
log(f"  After blended transfer discount (~70%): {score_impact*0.7:+.5f}")
log(f"  GATE: {'PASS ✅' if score_impact*0.7 >= 0.001 else 'FAIL ❌'}")

# ================================================================
# IF GATE PASSES: Apply to test and build submission
# ================================================================
if score_impact * 0.7 >= 0.001:
    log(f"\n{'='*60}")
    log("APPLYING CONSENSUS PRUNING TO TEST SET")
    log("="*60)
    
    # Load V4 as base
    import os
    v4_path = os.path.expanduser('~/Downloads/submission_v4_final.csv')
    v4 = pd.read_csv(v4_path)
    test_df_local = pd.read_csv('/home/mwaniasamuel/multilingual-health-qa/data/raw/Test.csv')
    
    pruned_answers = []
    changes_by_lang = defaultdict(int)
    total_by_lang = defaultdict(int)
    
    for i in tqdm(range(len(test_df_local)), desc="Pruning test"):
        rid = str(test_df_local.iloc[i]['ID'])
        sub = str(test_df_local.iloc[i]['subset'])
        total_by_lang[sub] += 1
        
        # Get current V4 answer
        v4_row = v4[v4['ID'] == rid]
        if len(v4_row) == 0:
            pruned_answers.append("No answer available.")
            continue
        
        answer = str(v4_row.iloc[0]['TargetR1F1']).strip()
        
        # Get candidates for this query (need test_cands or equivalent)
        try:
            cands = get_same_lang_candidates(
                str(test_df_local.iloc[i]['input']).strip(),
                test_emb[i], sub, k=5, exclude_exact=False
            )
        except:
            cands = [{'answer': answer}]
        
        if len(cands) < 2:
            pruned_answers.append(answer)
            continue
        
        # Consensus prune
        pruned = consensus_prune(answer, cands)
        if pruned != answer:
            changes_by_lang[sub] += 1
        pruned_answers.append(pruned)
    
    log(f"\nTest pruning changes by language:")
    for sub in sorted(total_by_lang.keys()):
        c = changes_by_lang.get(sub, 0)
        t = total_by_lang[sub]
        log(f"  {sub:<12} {c:>4}/{t:>4} changed ({100*c/t:.1f}%)")
    
    # Build submission
    sub_df = v4.copy()
    sub_df['TargetR1F1'] = pruned_answers
    sub_df['TargetRLF1'] = pruned_answers  # identical columns (safe)
    sub_df['TargetLLM'] = pruned_answers
    
    out_path = os.path.expanduser('~/Downloads/submission_v6_pruned.csv')
    sub_df.to_csv(out_path, index=False)
    log(f"\n✅ Saved: {out_path}")
    log(f"This is V4 + consensus sentence pruning. Identical columns.")
    log(f"Expected improvement: +{score_impact*0.7:.4f} on score (conservative)")
else:
    log("\n❌ Consensus pruning below threshold. Do NOT submit.")
    log("Lock V4+V2 as final selections.")
