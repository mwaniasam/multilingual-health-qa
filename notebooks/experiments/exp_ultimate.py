# =============================================================================
# EXPERIMENT ULTIMATE: Everything we missed, all at once
# Run AFTER bootstrap cell (needs: model, combined, val_df, test_df, 
# corpus_emb, val_emb, test_emb, val_cands_all, SUBSET_TO_LANG, etc.)
# =============================================================================
# 
# THREE UNTRIED APPROACHES IN ONE NOTEBOOK:
# 1. Q-A retrieval (index ANSWERS with 'passage:' prefix)
# 2. BM25 sparse retrieval (TF-IDF character n-grams)
# 3. Reciprocal Rank Fusion of ALL retrievers
# Plus: per-column MBR optimization
# =============================================================================

import numpy as np, pandas as pd, time, re, os, faiss, pickle
from collections import defaultdict, Counter
from tqdm import tqdm

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
UNI_RE = re.compile(r'\w+', re.UNICODE)
def uni_toks(text): return UNI_RE.findall(str(text).lower())

def uni_r1(ref_toks, hyp_toks):
    if not ref_toks or not hyp_toks: return 0.0
    rc, hc = Counter(ref_toks), Counter(hyp_toks)
    o = sum(min(rc[t], hc[t]) for t in rc)
    p, r = o/len(hyp_toks), o/len(ref_toks)
    return 2*p*r/(p+r) if p+r > 0 else 0.0

def uni_rl(ref_toks, hyp_toks):
    """Approximate ROUGE-L using LCS"""
    if not ref_toks or not hyp_toks: return 0.0
    m, n = len(ref_toks), len(hyp_toks)
    if m > 800 or n > 800:  # fallback for very long
        return uni_r1(ref_toks, hyp_toks) * 0.88
    prev = [0]*(n+1)
    for i in range(1, m+1):
        cur = [0]*(n+1)
        for j in range(1, n+1):
            cur[j] = prev[j-1]+1 if ref_toks[i-1]==hyp_toks[j-1] else max(prev[j], cur[j-1])
        prev = cur
    lcs = prev[n]
    p, r = lcs/n, lcs/m
    return 2*p*r/(p+r) if p+r > 0 else 0.0

TEST_MIX = {'Eng_Uga':0.284,'Aka_Gha':0.188,'Eng_Gha':0.188,'Lug_Uga':0.143,
            'Swa_Ken':0.087,'Eng_Ken':0.064,'Amh_Eth':0.023,'Eng_Eth':0.023}

# =============================================================================
# STEP 1: BUILD Q-A INDEX (Answer Embeddings with 'passage:' prefix)
# This is the big new idea — index ANSWERS, not questions
# E5 was DESIGNED for: query:"question" → passage:"answer"
# But we've been doing: query:"question" → query:"question"  
# =============================================================================
log("="*60)
log("STEP 1: Building Answer Embeddings (passage: prefix)")
log("This is what E5 was DESIGNED for — asymmetric search")
log("="*60)

import torch

corpus_answers = combined['output'].fillna('').tolist()

# Auto-detect encoding method from bootstrap
ans_emb_path = '/home/mwaniasamuel/multilingual-health-qa/data/processed/answer_embeddings.npy'
os.makedirs(os.path.dirname(ans_emb_path), exist_ok=True)

if os.path.exists(ans_emb_path):
    log(f"Loading cached answer embeddings from {ans_emb_path}")
    answer_emb = np.load(ans_emb_path)
    log(f"Loaded: {answer_emb.shape}")
else:
    log(f"Encoding {len(corpus_answers)} answers with 'passage: ' prefix...")
    
    # Try multiple encoding approaches
    try:
        # Method 1: If encode_texts function exists from bootstrap
        answer_emb = encode_texts(corpus_answers, prefix='passage: ')
        log("Used encode_texts() from bootstrap")
    except NameError:
        try:
            # Method 2: SentenceTransformer .encode()
            answer_texts = ['passage: ' + a for a in corpus_answers]
            answer_emb = model.encode(answer_texts, batch_size=64, 
                                       show_progress_bar=True, normalize_embeddings=True)
            log("Used SentenceTransformer .encode()")
        except (AttributeError, TypeError):
            # Method 3: Raw HuggingFace model + tokenizer
            answer_texts = ['passage: ' + a for a in corpus_answers]
            batch_size = 64
            all_embs = []
            for start in tqdm(range(0, len(answer_texts), batch_size), desc="Encoding answers"):
                batch = answer_texts[start:start+batch_size]
                with torch.no_grad():
                    encoded = tokenizer(batch, padding=True, truncation=True, 
                                      max_length=256, return_tensors='pt')
                    # Move to GPU if available
                    device = next(model.parameters()).device
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    outputs = model(**encoded)
                    mask = encoded['attention_mask'].unsqueeze(-1).float()
                    embs = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    all_embs.append(embs.cpu().numpy().astype('float32'))
            answer_emb = np.vstack(all_embs)
            log("Used raw HuggingFace model+tokenizer")
    
    answer_emb = answer_emb.astype('float32')
    np.save(ans_emb_path, answer_emb)
    log(f"Saved answer embeddings: {answer_emb.shape}")

# Build FAISS index for answers
log("Building FAISS index for answer embeddings...")
ans_index = faiss.IndexFlatIP(answer_emb.shape[1])
ans_index.add(answer_emb)
log(f"Answer index: {ans_index.ntotal} vectors")

# =============================================================================
# STEP 2: BUILD BM25 SPARSE INDEX
# Character n-gram TF-IDF for keyword matching
# =============================================================================
log(f"\n{'='*60}")
log("STEP 2: Building BM25/TF-IDF Sparse Index")
log("="*60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Build per-language TF-IDF indices
tfidf_indices = {}
for sub in SUBSET_TO_LANG.keys():
    sub_mask = combined['subset'] == sub
    sub_questions = combined[sub_mask]['input'].fillna('').tolist()
    sub_answers = combined[sub_mask]['output'].fillna('').tolist()
    sub_indices = np.where(sub_mask)[0]
    
    if len(sub_questions) < 10:
        continue
    
    # TF-IDF on questions (char n-grams for morphologically rich languages)
    tfidf_q = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5),
        max_features=50000, sublinear_tf=True
    ).fit(sub_questions)
    
    tfidf_q_matrix = tfidf_q.transform(sub_questions)
    
    # TF-IDF on answers too
    tfidf_a = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5),
        max_features=50000, sublinear_tf=True
    ).fit(sub_answers)
    
    tfidf_a_matrix = tfidf_a.transform(sub_answers)
    
    tfidf_indices[sub] = {
        'q_vectorizer': tfidf_q,
        'q_matrix': tfidf_q_matrix,
        'a_vectorizer': tfidf_a,
        'a_matrix': tfidf_a_matrix,
        'indices': sub_indices,
        'answers': sub_answers,
        'questions': sub_questions,
    }
    
    log(f"  {sub}: {len(sub_questions)} entries, Q-TF-IDF: {tfidf_q_matrix.shape}, A-TF-IDF: {tfidf_a_matrix.shape}")

# =============================================================================
# STEP 3: MULTI-RETRIEVER WITH RRF FUSION
# Combine: (1) Q-Q dense, (2) Q-A dense, (3) Q-Q sparse, (4) Q-A sparse
# =============================================================================
log(f"\n{'='*60}")
log("STEP 3: Multi-Retriever with RRF Fusion")
log("="*60)

def rrf_score(rankings, k=60):
    """Reciprocal Rank Fusion: combine multiple ranked lists."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

def get_candidates_multi(query_text, query_emb, subset, k_per_retriever=20, k_final=10):
    """Get candidates from ALL retrievers, fuse with RRF."""
    
    sub_mask = (combined['subset'] == subset).values
    sub_indices = np.where(sub_mask)[0]
    
    rankings = []
    
    # Retriever 1: Q-Q Dense (existing — query embedding vs question embeddings)
    if len(sub_indices) > 0:
        sub_q_emb = corpus_emb[sub_indices]
        sims = np.dot(sub_q_emb, query_emb)
        top_k_local = np.argsort(-sims)[:k_per_retriever]
        rankings.append([sub_indices[j] for j in top_k_local])
    
    # Retriever 2: Q-A Dense (NEW — query embedding vs answer embeddings)
    if len(sub_indices) > 0:
        sub_a_emb = answer_emb[sub_indices]
        sims = np.dot(sub_a_emb, query_emb)
        top_k_local = np.argsort(-sims)[:k_per_retriever]
        rankings.append([sub_indices[j] for j in top_k_local])
    
    # Retriever 3: Q-Q Sparse (TF-IDF question matching)
    if subset in tfidf_indices:
        ti = tfidf_indices[subset]
        q_vec = ti['q_vectorizer'].transform([query_text])
        sims = cosine_similarity(q_vec, ti['q_matrix'])[0]
        top_k_local = np.argsort(-sims)[:k_per_retriever]
        rankings.append([ti['indices'][j] for j in top_k_local])
    
    # Retriever 4: Q-A Sparse (TF-IDF question → answer matching)
    if subset in tfidf_indices:
        ti = tfidf_indices[subset]
        # Use question vectorizer on query, search against answer TF-IDF
        # This requires transforming query with answer vocabulary
        q_vec = ti['a_vectorizer'].transform([query_text])
        sims = cosine_similarity(q_vec, ti['a_matrix'])[0]
        top_k_local = np.argsort(-sims)[:k_per_retriever]
        rankings.append([ti['indices'][j] for j in top_k_local])
    
    # RRF Fusion
    fused = rrf_score(rankings)
    
    # Build candidate list
    candidates = []
    seen = set()
    for idx, score in fused[:k_final]:
        if idx in seen:
            continue
        seen.add(idx)
        candidates.append({
            'idx': int(idx),
            'answer': str(combined.iloc[idx]['output']).strip(),
            'question': str(combined.iloc[idx]['input']).strip(),
            'subset': str(combined.iloc[idx]['subset']),
            'rrf_score': score,
        })
    
    return candidates

# =============================================================================
# STEP 4: PER-COLUMN MBR SELECTION
# R1-MBR for TargetR1F1, RL-MBR for TargetRLF1, Comprehensive for TargetLLM
# =============================================================================
log(f"\n{'='*60}")
log("STEP 4: Per-Column MBR Selection")
log("="*60)

def mbr_select(candidates, metric_fn):
    """Select candidate with highest expected metric against all others."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]['answer']
    
    best_idx = 0
    best_score = -1
    
    for i, c in enumerate(candidates):
        c_toks = uni_toks(c['answer'])
        if not c_toks:
            continue
        scores = []
        for j, other in enumerate(candidates):
            if i == j:
                continue
            o_toks = uni_toks(other['answer'])
            if o_toks:
                scores.append(metric_fn(o_toks, c_toks))
        
        avg = np.mean(scores) if scores else 0
        if avg > best_score:
            best_score = avg
            best_idx = i
    
    return candidates[best_idx]['answer']

def select_comprehensive(candidates):
    """Select the most comprehensive answer for LLM judge.
    Prefer: longest answer with highest vocabulary diversity."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]['answer']
    
    best_idx = 0
    best_score = -1
    
    for i, c in enumerate(candidates):
        c_toks = uni_toks(c['answer'])
        unique_ratio = len(set(c_toks)) / max(len(c_toks), 1)
        # Balance length and diversity
        score = len(c_toks) * unique_ratio * (c.get('rrf_score', 0.1) + 0.01)
        if score > best_score:
            best_score = score
            best_idx = i
    
    return candidates[best_idx]['answer']

# =============================================================================
# STEP 5: VALIDATE ON VAL
# Compare: old pipeline vs new multi-retriever + per-column MBR
# =============================================================================
log(f"\n{'='*60}")
log("STEP 5: VALIDATION — Old vs New Pipeline")
log("="*60)

old_results = defaultdict(lambda: {'r1': [], 'rl': []})
new_results = defaultdict(lambda: {'r1': [], 'rl': [], 'r1_mbr': [], 'rl_mbr': []})

for i in tqdm(range(len(val_df)), desc="Validating"):
    sub = str(val_df.iloc[i]['subset'])
    ref = str(val_df.iloc[i]['output']).strip()
    query = str(val_df.iloc[i]['input']).strip()
    rt = uni_toks(ref)
    if not rt:
        continue
    
    # Old pipeline (existing top-1)
    try:
        old_ans = val_cands_all[i][0]['answer'] if val_cands_all[i] else ""
    except:
        old_ans = ""
    old_r1 = uni_r1(rt, uni_toks(old_ans))
    old_results[sub]['r1'].append(old_r1)
    
    # New pipeline: multi-retriever + RRF
    new_cands = get_candidates_multi(query, val_emb[i], sub, k_per_retriever=20, k_final=10)
    
    if not new_cands:
        new_results[sub]['r1'].append(old_r1)
        new_results[sub]['r1_mbr'].append(old_r1)
        continue
    
    # New top-1 (RRF-fused)
    new_top1 = new_cands[0]['answer']
    new_r1 = uni_r1(rt, uni_toks(new_top1))
    new_results[sub]['r1'].append(new_r1)
    
    # Per-column MBR
    r1_answer = mbr_select(new_cands, uni_r1)
    new_r1_mbr = uni_r1(rt, uni_toks(r1_answer))
    new_results[sub]['r1_mbr'].append(new_r1_mbr)

# Print comparison
log(f"\n{'Sub':<12} {'Old R1':>8} {'New Top1':>8} {'New MBR':>8} {'Δ Top1':>8} {'Δ MBR':>8}")
log("-"*60)

tw_old, tw_new_t1, tw_new_mbr = 0, 0, 0
for sub in sorted(SUBSET_TO_LANG.keys()):
    o = np.mean(old_results[sub]['r1']) if old_results[sub]['r1'] else 0
    n1 = np.mean(new_results[sub]['r1']) if new_results[sub]['r1'] else 0
    nm = np.mean(new_results[sub]['r1_mbr']) if new_results[sub]['r1_mbr'] else 0
    w = TEST_MIX.get(sub, 0)
    tw_old += w*o; tw_new_t1 += w*n1; tw_new_mbr += w*nm
    d1 = n1 - o; dm = nm - o
    m1 = " ★" if d1 > 0.003 else (" ⚠️" if d1 < -0.003 else "")
    mm = " ★" if dm > 0.003 else (" ⚠️" if dm < -0.003 else "")
    log(f"  {sub:<12} {o:>8.4f} {n1:>8.4f} {nm:>8.4f} {d1:>+8.4f}{m1} {dm:>+8.4f}{mm}")

log(f"\n  Test-weighted R1:")
log(f"    Old pipeline:     {tw_old:.4f}")
log(f"    New RRF top-1:    {tw_new_t1:.4f} ({tw_new_t1-tw_old:+.5f})")
log(f"    New RRF + MBR:    {tw_new_mbr:.4f} ({tw_new_mbr-tw_old:+.5f})")

gate = (tw_new_mbr - tw_old) * 0.37
log(f"\n  Estimated score impact: {gate:+.5f}")
log(f"  GATE: {'PASS ✅' if gate >= 0.001 else 'FAIL ❌'}")

# =============================================================================
# STEP 6: IF GATE PASSES — Build Test Submission
# =============================================================================
if gate >= 0.001:
    log(f"\n{'='*60}")
    log("STEP 6: Building Test Submission with 3-column optimization")
    log("="*60)
    
    test_r1_answers = []
    test_rl_answers = []
    test_llm_answers = []
    
    for i in tqdm(range(len(test_df)), desc="Building test submission"):
        query = str(test_df.iloc[i]['input']).strip()
        sub = str(test_df.iloc[i]['subset'])
        
        cands = get_candidates_multi(query, test_emb[i], sub, k_per_retriever=20, k_final=10)
        
        if not cands:
            fallback = "No answer available."
            test_r1_answers.append(fallback)
            test_rl_answers.append(fallback)
            test_llm_answers.append(fallback)
            continue
        
        # Per-column optimization
        r1_ans = mbr_select(cands, uni_r1)
        rl_ans = mbr_select(cands, uni_rl)  # RL-specific MBR
        llm_ans = select_comprehensive(cands)  # Most comprehensive for judge
        
        test_r1_answers.append(r1_ans)
        test_rl_answers.append(rl_ans)
        test_llm_answers.append(llm_ans)
    
    # Build submission DataFrame
    sub_df = pd.DataFrame({
        'ID': test_df['ID'],
        'TargetR1F1': test_r1_answers,
        'TargetRLF1': test_rl_answers,
        'TargetLLM': test_llm_answers,
    })
    
    # Check how many entries have per-column splits
    n_r1_neq_rl = sum(1 for a, b in zip(test_r1_answers, test_rl_answers) if a != b)
    n_r1_neq_llm = sum(1 for a, b in zip(test_r1_answers, test_llm_answers) if a != b)
    
    log(f"\n  R1≠RL: {n_r1_neq_rl}/{len(test_df)} ({100*n_r1_neq_rl/len(test_df):.1f}%)")
    log(f"  R1≠LLM: {n_r1_neq_llm}/{len(test_df)} ({100*n_r1_neq_llm/len(test_df):.1f}%)")
    
    # Save per-column submission
    out_split = os.path.expanduser('~/Downloads/submission_v7_ultimate_split.csv')
    sub_df.to_csv(out_split, index=False)
    log(f"  Saved SPLIT: {out_split}")
    
    # Also save identical-column version (safe fallback)
    sub_df_id = pd.DataFrame({
        'ID': test_df['ID'],
        'TargetR1F1': test_r1_answers,
        'TargetRLF1': test_r1_answers,
        'TargetLLM': test_r1_answers,
    })
    out_id = os.path.expanduser('~/Downloads/submission_v7_ultimate_identical.csv')
    sub_df_id.to_csv(out_id, index=False)
    log(f"  Saved IDENTICAL: {out_id}")
    
    log(f"\n✅ DONE! Submit the IDENTICAL version first (safer).")
    log(f"   If it improves, try the SPLIT version next.")
else:
    log("\n⚠️ New pipeline below threshold on val.")
    log("Falling back: try just the RRF candidates with existing MBR...")
    
    # Even if overall gate fails, check if specific languages improved
    log("\nPer-language: which languages improved?")
    for sub in sorted(SUBSET_TO_LANG.keys()):
        o = np.mean(old_results[sub]['r1']) if old_results[sub]['r1'] else 0
        nm = np.mean(new_results[sub]['r1_mbr']) if new_results[sub]['r1_mbr'] else 0
        if nm > o + 0.005:
            log(f"  ★ {sub}: {o:.4f} → {nm:.4f} (+{nm-o:.4f}) — use new pipeline for this language")
