"""
build_submissions.py
====================
The submission builders, recovered verbatim. Run after bootstrap + Section A reload
(training_cells.py). V7 is the best file and a selected final; V4 is the second final.

  V4  -> submission_v4_final.csv   (0.6898, selected final / hedge)
  V6  -> submission_v6.csv         (intermediate stack)
  V7  -> submission_v7.csv         (0.6908 BEST, selected final)  = V6 + Amh QA-union
  SIM -> test-mix-weighted local leaderboard simulator

All files use IDENTICAL columns and are fully open-source.
"""

# ============================================================================
# V4  [SELECTED FINAL — hedge]
# V3 winners + embedding interpolation (Eng_Uga/Lug_Uga/Swa_Ken) + Aka re-decision.
# ============================================================================
BUILD_V4 = r'''
import json, numpy as np, pandas as pd
from tqdm import tqdm
ft2_corpus = np.load(CACHE/'ft2_corpus.npy'); ft2_test = np.load(CACHE/'ft2_test.npy')
INTERP = {'Eng_Uga': 0.6, 'Lug_Uga': 0.6, 'Swa_Ken': 0.9}   # adopted betas

def interp_pool_test(i, sub, beta, k=K_CANDIDATES):
    _, mask = lang_indices[sub]; mask = np.array(mask)
    s = beta*(corpus_emb[mask] @ test_emb[i]) + (1-beta)*(ft2_corpus[mask] @ ft2_test[i])
    order = np.argsort(-s); out = []
    for j in order[:k+5]:
        ci = int(mask[j])
        out.append({'answer': answers_raw[ci], 'sim': float(s[j]), 'idx': ci})
        if len(out) >= k: break
    return out   # exclude_exact=False behavior at test

# Aka_Gha re-decision: stitch vs comb_mbr on FULL weighted holdout
sub = 'Aka_Gha'; tag, a, m = choice[sub]
idxs = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset'])==sub
        and val_cands_all[i] and str(val_df.iloc[i]['output']).strip()]
hold = idxs[1::2]; s_st, s_cb = [], []
for i in hold:
    rt = uni_toks(str(val_df.iloc[i]['output']))[:CAP]
    st = uni_toks(uni_stitch(val_cands_all[i], 0.70, sub))[:CAP]
    dd, ddw, u1, uL = uni_prep(val_cands_all[i])
    cb = uni_toks(dd[mbr_idx(0.5*u1+0.5*uL, ddw, a, m)])[:CAP]
    s_st.append(0.37*uni_r1(rt, st) + 0.37*uni_rl(rt, st))
    s_cb.append(0.37*uni_r1(rt, cb) + 0.37*uni_rl(rt, cb))
AKA_PICK = 'stitch' if np.mean(s_st) > np.mean(s_cb) + 0.002 else 'comb_mbr'
strategy = json.load(open(OUTPUT_DIR/'v3_strategy.json')); strategy['Aka_Gha'] = AKA_PICK
ft_test_ans = json.load(open(OUTPUT_DIR/'ftqwen_test_answers.json'))

rows = []
for i in tqdm(range(len(test_df)), desc="V4 assemble"):
    rid = str(test_df.iloc[i]['ID']); sub = test_subs[i]; tag, a, m = choice.get(sub, ('2leg',0.05,99.0))
    if sub in INTERP:
        pool = interp_pool_test(i, sub, INTERP[sub])
    elif tag == '4leg':
        pool = union4(test_qs[i].strip(), test_emb[i], gem_test[i], bge_test[i], sub, exclude_exact=False)
    else:
        pool = get_same_lang_candidates(test_qs[i].strip(), test_emb[i], sub, k=K_CANDIDATES, exclude_exact=False)
    if not pool:
        rows.append({'ID': test_df.iloc[i]['ID'],'TargetR1F1':'No answer','TargetRLF1':'No answer','TargetLLM':'No answer'}); continue
    pick = strategy.get(sub, 'comb_mbr'); gen = ft_test_ans.get(rid, '')
    if pick == 'ft_gen' and gen: ans = gen
    elif pick == 'stitch':       ans = uni_stitch(pool, uni_stitch_gate[sub]['lam'], sub)
    else:
        dd, ddw, u1, uL = uni_prep(pool); ans = dd[mbr_idx(0.5*u1+0.5*uL, ddw, a, m)]
    rows.append({'ID': test_df.iloc[i]['ID'],'TargetR1F1':ans,'TargetRLF1':ans,'TargetLLM':ans})
df = pd.DataFrame(rows).reindex(columns=SUB_COLS)
assert len(df) == len(test_df) and not df.isnull().any().any()
df.to_csv(OUTPUT_DIR/'submission_v4_final.csv', index=False); print("Saved: submission_v4_final.csv")
'''

# ============================================================================
# V6  [intermediate stack]
# V4 + CE-base-stitch (Aka_Gha, Amh_Eth) + Lug_Uga adapter beta=0.8 + Swa_Ken beta=0.8.
# Requires cmod/ctok (ce_scores) loaded.
# ============================================================================
BUILD_V6 = r'''
import json, numpy as np, pandas as pd, torch
from tqdm import tqdm
ft2_corpus = np.load(CACHE/'ft2_corpus.npy'); ft2_test = np.load(CACHE/'ft2_test.npy')
pl_lug = np.load(CACHE/'pl_Lug_Uga_test.npy'); lug_corpus = np.load(CACHE/'pl_Lug_Uga_corpus.npy')
pl_lug_idx = json.load(open(CACHE/'pl_Lug_Uga_test_idx.json')); pl_lug_map = {i:j for j,i in enumerate(pl_lug_idx)}
CE_LANGS = {'Aka_Gha', 'Amh_Eth'}
v4 = pd.read_csv(OUTPUT_DIR/'submission_v4_final.csv'); v4_map = dict(zip(v4['ID'].astype(str), v4['TargetR1F1']))

rows = []
for i in tqdm(range(len(test_df)), desc="V6"):
    rid = str(test_df.iloc[i]['ID']); sub = test_subs[i]
    if sub in CE_LANGS:
        pool = get_same_lang_candidates(test_qs[i].strip(), test_emb[i], sub, k=K_CANDIDATES, exclude_exact=False)
        if pool:
            cs = ce_scores(test_qs[i], [questions_raw[c['idx']] for c in pool]); order = np.argsort(-cs)
            cp = [{'answer':pool[j]['answer'],'sim':float(cs[j]),'idx':pool[j]['idx']} for j in order]
            lam = uni_stitch_gate[sub]['lam'] if sub in uni_stitch_gate else 0.70
            ans = uni_stitch(cp, lam, sub)
        else: ans = v4_map[rid]
    elif sub == 'Lug_Uga':
        _, mask = lang_indices[sub]; mask_arr = np.array(mask)
        s = 0.8*(corpus_emb[mask_arr] @ test_emb[i]) + 0.2*(lug_corpus @ pl_lug[pl_lug_map[i]])
        order = np.argsort(-s)
        pool = [{'answer':answers_raw[int(mask_arr[j])],'sim':float(s[j]),'idx':int(mask_arr[j])} for j in order[:K_CANDIDATES]]
        tag, a, m = choice[sub]; dd, ddw, u1, uL = uni_prep(pool); ans = dd[mbr_idx(0.5*u1+0.5*uL, ddw, a, m)]
    elif sub == 'Swa_Ken':
        _, mask = lang_indices[sub]; mask_arr = np.array(mask)
        s = 0.8*(corpus_emb[mask_arr] @ test_emb[i]) + 0.2*(ft2_corpus[mask_arr] @ ft2_test[i])
        order = np.argsort(-s)
        pool = [{'answer':answers_raw[int(mask_arr[j])],'sim':float(s[j]),'idx':int(mask_arr[j])} for j in order[:K_CANDIDATES]]
        tag, a, m = choice[sub]; dd, ddw, u1, uL = uni_prep(pool); ans = dd[mbr_idx(0.5*u1+0.5*uL, ddw, a, m)]
    else:
        ans = v4_map[rid]
    rows.append({'ID': test_df.iloc[i]['ID'],'TargetR1F1':ans,'TargetRLF1':ans,'TargetLLM':ans})
df = pd.DataFrame(rows).reindex(columns=SUB_COLS)
assert len(df) == len(test_df) and not df.isnull().any().any()
df.to_csv(OUTPUT_DIR/'submission_v6.csv', index=False); print("Saved: submission_v6.csv")
'''

# ============================================================================
# V7  [SELECTED FINAL — BEST 0.6908]
# V6 + Amh_Eth QA-union CE-stitch (only ~60 rows change). Requires enc_model + ce_scores.
# ============================================================================
BUILD_V7 = r'''
import numpy as np, pandas as pd, torch
from tqdm import tqdm
sub = 'Amh_Eth'
_, mask = lang_indices[sub]; mask_arr = np.array(mask)
qa_texts = [f"passage: {questions_raw[ci]} {answers_raw[ci]}" for ci in mask_arr]
qa_emb = enc_model.encode(qa_texts, batch_size=32, normalize_embeddings=True,
                          show_progress_bar=True).astype(np.float32)
np.save(CACHE/'qa_emb_amh.npy', qa_emb)
g_lam = uni_stitch_gate.get(sub, {'lam': 0.70})['lam']
t_idx = [i for i in range(len(test_df)) if test_subs[i] == sub]; new_ans = {}
for i in tqdm(t_idx, desc="Amh V7"):
    pool = get_same_lang_candidates(test_qs[i].strip(), test_emb[i], sub, k=K_CANDIDATES, exclude_exact=False)
    s = qa_emb @ test_emb[i]
    qa_pool = [{'answer':answers_raw[int(mask_arr[j])],'sim':float(s[j]),'idx':int(mask_arr[j])} for j in np.argsort(-s)[:15]]
    seen = {c['idx'] for c in pool}
    upool = pool + [c for c in qa_pool if c['idx'] not in seen]
    if not upool: continue
    cs = ce_scores(test_qs[i], [questions_raw[c['idx']] for c in upool])
    cp = [{'answer':upool[j]['answer'],'sim':float(cs[j]),'idx':upool[j]['idx']} for j in np.argsort(-cs)]
    new_ans[str(test_df.iloc[i]['ID'])] = uni_stitch(cp, g_lam, sub)
v6 = pd.read_csv(OUTPUT_DIR/'submission_v6.csv')
sel = v6['ID'].astype(str).isin(new_ans)
for col in ['TargetR1F1','TargetRLF1','TargetLLM']:
    v6.loc[sel, col] = v6.loc[sel, 'ID'].astype(str).map(new_ans)
assert len(v6) == len(test_df) and not v6.isnull().any().any()
v6.to_csv(OUTPUT_DIR/'submission_v7.csv', index=False)
print(f"Rows changed: {sel.sum()}  -> Saved: submission_v7.csv (BEST)")
'''

# ============================================================================
# LB SIMULATOR — test-mix-weighted local score (runs ~+0.005 optimistic vs LB)
# ============================================================================
LB_SIMULATOR = r'''
test_mix = test_df['subset'].value_counts(normalize=True).to_dict()
sim_r1 = sim_rl = 0.0
for sub in sorted(SUBSET_TO_LANG):
    tag, a, m = choice[sub]; pr, rf = P[tag]['prep'], P[tag]['ref']
    idxs = [i for i in range(len(val_df)) if str(val_df.iloc[i]['subset'])==sub and rf[i]]
    g = uni_stitch_gate.get(sub, {})
    if g.get('use'):
        pools = val_cands_all if g['pool']=='2leg' else v4c
        r1 = np.mean([uni_r1(uni_toks(str(val_df.iloc[i]['output']))[:CAP],
              uni_toks(uni_stitch(pools[i], g['lam'], sub))[:CAP]) for i in idxs[1::2]])
    else:
        r1 = np.mean([rf[i][mbr_idx(pr[i][2], pr[i][1], a, m)][0] for i in idxs])
    rl = np.mean([rf[i][mbr_idx(pr[i][3], pr[i][1], a, m)][1] for i in idxs])
    w = test_mix.get(sub, 0); sim_r1 += w*r1; sim_rl += w*rl
print(f"TEST-WEIGHTED sim: R1={sim_r1:.4f} RL={sim_rl:.4f}")
print(f"Weighted score (LLM@0.82): {0.37*sim_r1+0.37*sim_rl+0.26*0.82:.4f}")
'''

if __name__ == "__main__":
    print("Builders: V4 (final hedge), V6 (intermediate), V7 (BEST final), + LB simulator.")
