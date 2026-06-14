# ================================================================
# LAST ATTEMPT: Query-based sentence pruning
# Remove sentences from answer that have ZERO overlap with the question
# ================================================================
import re, time, numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
UNI_RE = re.compile(r'\w+', re.UNICODE)
SENT_RE = re.compile(r'(?<=[.!?።\n])\s+')
STOPWORDS = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
             'do','does','did','will','would','shall','should','may','might','can','could',
             'and','or','but','if','of','in','on','at','to','for','with','by','from','as',
             'this','that','these','those','it','its','they','them','their','we','our','you',
             'your','i','my','me','he','she','his','her','what','which','who','how','when',
             'where','why','not','no','yes','so','very','just','also','about','than','then',
             'into','over','after','before','between','under','through','during','against',
             'na','ya','wa','ni','kwa','la','za','au','je','si','no','ngo','mu','oku','ba',
             'ne','wo','no','a','ɛ','sɛ','ho','de','wɔ','bɛ','ɛno','ɛna','ane','na','nso'}

def content_toks(text):
    """Get content tokens (non-stopword) from text."""
    return [t for t in UNI_RE.findall(str(text).lower()) if t not in STOPWORDS and len(t) > 2]

def split_sents(text):
    sents = SENT_RE.split(str(text).strip())
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]

def uni_toks_l(text): return UNI_RE.findall(str(text).lower())

def uni_r1_l(ref_toks, hyp_toks):
    if not ref_toks or not hyp_toks: return 0.0
    rc = Counter(ref_toks); hc = Counter(hyp_toks)
    overlap = sum(min(rc[t], hc[t]) for t in rc)
    p, r = overlap/len(hyp_toks), overlap/len(ref_toks)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

try: uni_toks; uni_r1
except: uni_toks = uni_toks_l; uni_r1 = uni_r1_l

TEST_MIX = {'Eng_Uga':0.284,'Aka_Gha':0.188,'Eng_Gha':0.188,'Lug_Uga':0.143,
            'Swa_Ken':0.087,'Eng_Ken':0.064,'Amh_Eth':0.023,'Eng_Eth':0.023}

# ================================================================
# Method A: Remove sentences with zero content-word query overlap
# ================================================================
def query_prune_zero_overlap(answer, query):
    """Remove sentences that share zero content words with the query."""
    sents = split_sents(answer)
    if len(sents) <= 1:
        return answer
    
    q_content = set(content_toks(query))
    if not q_content:
        return answer
    
    kept = []
    for s in sents:
        s_content = set(content_toks(s))
        if s_content & q_content:  # at least one shared content word
            kept.append(s)
    
    if not kept:
        return answer  # safety: keep everything if all would be removed
    
    return ' '.join(kept)

# ================================================================
# Method B: Greedy removal using QUERY as pseudo-reference  
# ================================================================
def query_prune_greedy(answer, query):
    """Greedily remove sentences that improve F1 overlap with query."""
    sents = split_sents(answer)
    if len(sents) <= 1:
        return answer
    
    qt = uni_toks(query)
    if not qt:
        return answer
    
    at = uni_toks(answer)
    full_score = uni_r1(qt, at)
    
    current_sents = list(sents)
    best_score = full_score
    best_text = answer
    
    improved = True
    while improved and len(current_sents) > 1:
        improved = False
        best_removal = -1
        for j in range(len(current_sents)):
            candidate = ' '.join(current_sents[:j] + current_sents[j+1:])
            ct = uni_toks(candidate)
            if not ct: continue
            score = uni_r1(qt, ct)
            if score > best_score + 0.005:
                best_score = score
                best_removal = j
                best_text = candidate
                improved = True
        if best_removal >= 0:
            current_sents = current_sents[:best_removal] + current_sents[best_removal+1:]
    
    return best_text

# ================================================================
# Method C: Keep only top-N sentences by query relevance
# ================================================================
def query_prune_topn(answer, query, keep_ratio=0.75):
    """Keep only the most query-relevant sentences."""
    sents = split_sents(answer)
    if len(sents) <= 2:
        return answer
    
    qt = set(content_toks(query))
    if not qt:
        return answer
    
    scored = []
    for s in sents:
        st = set(content_toks(s))
        overlap = len(st & qt)
        scored.append((overlap, s))
    
    # Keep at least keep_ratio of sentences, always keeping those with overlap > 0
    n_keep = max(1, int(len(sents) * keep_ratio))
    scored.sort(key=lambda x: -x[0])
    
    # But preserve original order
    kept_sents = set(s for _, s in scored[:n_keep])
    result = [s for s in sents if s in kept_sents]
    
    return ' '.join(result) if result else answer

# ================================================================
# EVALUATE ALL METHODS ON VAL
# ================================================================
log("="*60)
log("QUERY-BASED PRUNING — 3 Methods on Val")
log("="*60)

methods = {
    'A: zero-overlap': query_prune_zero_overlap,
    'B: greedy-query':  query_prune_greedy,
    'C: top-75%':       lambda a, q: query_prune_topn(a, q, 0.75),
}

for method_name, prune_fn in methods.items():
    results = defaultdict(lambda: {'before':[], 'after':[], 'changed':0})
    
    for i in tqdm(range(len(val_df)), desc=method_name):
        sub = str(val_df.iloc[i]['subset'])
        ref = str(val_df.iloc[i]['output']).strip()
        query = str(val_df.iloc[i]['input']).strip()
        rt = uni_toks(ref)
        if not rt: continue
        
        try:
            if not val_cands_all[i]: continue
            answer = val_cands_all[i][0]['answer']
        except: continue
        
        at = uni_toks(answer)
        if not at: continue
        
        before = uni_r1(rt, at)
        pruned = prune_fn(answer, query)
        pt = uni_toks(pruned)
        after = uni_r1(rt, pt) if pt else before
        
        results[sub]['before'].append(before)
        results[sub]['after'].append(after)
        if pruned != answer:
            results[sub]['changed'] += 1
    
    log(f"\n--- {method_name} ---")
    tw_b, tw_a = 0, 0
    for sub in sorted(SUBSET_TO_LANG.keys()):
        r = results[sub]
        b = np.mean(r['before']) if r['before'] else 0
        a = np.mean(r['after']) if r['after'] else 0
        w = TEST_MIX.get(sub, 0)
        tw_b += w*b; tw_a += w*a
        n = len(r['before']); c = r['changed']
        d = a - b
        marker = " ★" if d > 0.003 else (" ⚠️" if d < -0.003 else "")
        log(f"  {sub:<12} {b:.4f} → {a:.4f}  Δ={d:+.4f}  chg={c}/{n}{marker}")
    
    delta = tw_a - tw_b
    log(f"  Weighted: {tw_b:.4f} → {tw_a:.4f}  Δ={delta:+.5f}  score_impact={delta*0.37:+.5f}")

log(f"\n{'='*60}")
log("DONE. Compare methods. If any shows positive delta, build test submission.")
log("If all negative or zero, the predecessor was right. Lock V4+V2.")
