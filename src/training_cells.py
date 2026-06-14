"""
training_cells.py
=================
Every model-training cell, in dependency order, recovered verbatim from the sessions.
Run each as a Colab cell AFTER the bootstrap cell (see bootstrap_contract.py for the
objects each one needs). Sections:

  A. Model reload prep (reattach AfriE5 + CE + Qwen in a fresh session)
  B. Cross-encoder reranker v2  (THE reranker used by V6/V7)  [ADOPTED]
  C. FT2 global retriever fine-tune (used only via interpolation) [ADOPTED-as-interp]
  D. Per-language retriever adapter (Lug_Uga, answer-similarity)   [ADOPTED]
  E. Pseudo-labeling the CE (the last experiment; not completed)   [UNTRIED]
  F. Qwen2.5-7B LoRA resume config (epoch settings reference)      [epoch1 ADOPTED]

All training mines/fine-tunes on TRAIN rows only; val is the judge. Seeds = 42.
"""

# ============================================================================
# A. MODEL RELOAD PREP — fresh session: pip line -> bootstrap -> this cell
#    !pip install -q -U torchao pylcs bitsandbytes transformers accelerate
# ============================================================================
RELOAD = r'''
import torch, gc
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM)
from peft import PeftModel
gc.collect(); torch.cuda.empty_cache()

# 1) AfriE5 encoder (Q->Q backbone; also encodes qa_emb legs)
enc_model = SentenceTransformer(str(OUTPUT_DIR/'afrie5-final-model'), device='cuda:0')
enc_model.max_seq_length = 256

# 2) CE reranker (feeds the stitch on Aka_Gha, Amh_Eth)
ctok = AutoTokenizer.from_pretrained(str(OUTPUT_DIR/'ce-reranker-v2'))
cmod = AutoModelForSequenceClassification.from_pretrained(
    str(OUTPUT_DIR/'ce-reranker-v2')).to('cuda:0').eval()
@torch.no_grad()
def ce_scores(query, cand_qs):
    enc = ctok([query]*len(cand_qs), cand_qs, padding=True, truncation=True,
               max_length=160, return_tensors='pt').to('cuda:0')
    return torch.softmax(cmod(**enc).logits, -1)[:, 1].cpu().numpy()

# 3) FT Qwen (Eng_Gha generation)
FT_MODEL_DIR = OUTPUT_DIR / 'qwen-ft-health'
tok = AutoTokenizer.from_pretrained(str(FT_MODEL_DIR))
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',
        dtype=torch.bfloat16, device_map='cuda:0')
ft = PeftModel.from_pretrained(base, str(FT_MODEL_DIR)); ft.eval()
@torch.no_grad()
def ft_generate(q, lang, cands, max_new=350):
    ctx = "\n".join(f"{k+1}. {c['answer']}" for k, c in enumerate(cands[:5]))
    msgs = [{"role":"system","content":
             f"You are a multilingual health expert. Answer health questions based on the "
             f"reference information provided. Use the EXACT words and phrases from the "
             f"references when possible. Be complete and accurate. Answer in {lang}."},
            {"role":"user","content": f"Question: {q}\n\nReference answers:\n{ctx}"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors='pt', return_dict=True).to('cuda:0')
    out = ft.generate(**enc, max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print(f"all loaded - free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")  # ~18GB used on 24GB L4
'''

# ============================================================================
# B. CROSS-ENCODER RERANKER v2  [ADOPTED — used by V6/V7]
#    xlm-roberta-base, binary classification, unicode labels, guarded deployment.
# ============================================================================
CE_RERANKER_V2 = r'''
import torch, gc, random, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
random.seed(42); np.random.seed(42)

# ---- 1) MINE PAIRS (unicode labels, train rows only, all languages) ----
train_q_set = set(q.strip() for q in train_df['input'].dropna().astype(str))
is_train_row = np.array([q in train_q_set for q in corpus_q_stripped])
pairs = []   # (query_text, cand_question_text, label)
PER_LANG = 2500
for sub in SUBSET_TO_LANG:
    idx_t = [i for i in range(len(combined)) if subsets_raw[i]==sub and is_train_row[i]]
    random.shuffle(idx_t)
    ix, mask = lang_indices[sub]; mask_arr = np.array(mask)
    tmask = np.array([is_train_row[ci] for ci in mask])
    made = 0
    for qi in idx_t:
        if made >= PER_LANG: break
        gold = uni_toks(answers_raw[qi])[:CAP]
        if len(gold) < 3: continue
        D, I = ix.search(corpus_emb[qi].reshape(1,-1), 20)
        pos, neg = [], []
        for li in I[0]:
            if li < 0 or not tmask[int(li)]: continue
            ci = int(mask_arr[int(li)])
            if ci == qi or corpus_q_stripped[ci] == corpus_q_stripped[qi]: continue
            r1 = uni_r1(gold, uni_toks(answers_raw[ci])[:CAP])
            (pos if r1 >= 0.5 else neg if r1 <= 0.2 else []).append(ci)
        if pos and neg:
            pairs.append((questions_raw[qi], questions_raw[random.choice(pos)], 1))
            for n_ in random.sample(neg, min(2, len(neg))):
                pairs.append((questions_raw[qi], questions_raw[n_], 0))
            made += 1
print(f"Pairs: {len(pairs)}  (pos {sum(1 for p in pairs if p[2]==1)})")

# ---- 2) TRAIN xlm-roberta-base cross-encoder (binary) ----
gc.collect(); torch.cuda.empty_cache()
CE = 'xlm-roberta-base'
ctok = AutoTokenizer.from_pretrained(CE)
cmod = AutoModelForSequenceClassification.from_pretrained(CE, num_labels=2).to('cuda:0')
opt = torch.optim.AdamW(cmod.parameters(), lr=2e-5)
random.shuffle(pairs); B = 16
cmod.train()
for ep in range(1):
    pbar = tqdm(range(0, len(pairs), B), desc=f"CE epoch {ep+1}")
    for s in pbar:
        chunk = pairs[s:s+B]
        enc = ctok([p[0] for p in chunk], [p[1] for p in chunk], padding=True,
                   truncation=True, max_length=160, return_tensors='pt').to('cuda:0')
        labels = torch.tensor([p[2] for p in chunk]).to('cuda:0')
        out = cmod(**enc, labels=labels)
        out.loss.backward(); opt.step(); opt.zero_grad()
        if s % (B*50) == 0: pbar.set_postfix(loss=float(out.loss))
cmod.eval()
cmod.save_pretrained(str(OUTPUT_DIR/'ce-reranker-v2'))
ctok.save_pretrained(str(OUTPUT_DIR/'ce-reranker-v2'))
# Deploy: rerank top-15 by softmax[:,1], feed CE-ordered pool to uni_stitch on Aka/Amh.
'''

# ============================================================================
# C. FT2 GLOBAL RETRIEVER FINE-TUNE  [used only via interpolation]
#    1 careful MNRL round on train-mined hard triplets. Catastrophic forgetting if
#    deployed directly; salvaged as beta*AfriE5 + (1-beta)*FT2 interpolation.
# ============================================================================
FT2_RETRIEVER = r'''
import torch, gc, random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
random.seed(42); np.random.seed(42)
AFRIE5_DIR = OUTPUT_DIR / 'afrie5-final-model'; FT2_DIR = OUTPUT_DIR / 'afrie5-ft2'
PREFIX = "query: "
train_q_set = set(q.strip() for q in train_df['input'].dropna().astype(str))
is_train_row = np.array([corpus_q_stripped[i] in train_q_set for i in range(len(combined))])

MINE_CAP = {'Eng_Uga':4000,'Lug_Uga':2500,'Aka_Gha':2500,'Eng_Gha':2500,
            'Swa_Ken':1200,'Eng_Ken':1200,'Eng_Eth':800,'Amh_Eth':600}
triplets = []
for sub, cap in MINE_CAP.items():
    idx_t = [i for i in range(len(combined)) if subsets_raw[i]==sub and is_train_row[i]]
    random.shuffle(idx_t); idx_t = idx_t[:cap]
    idx_map, mask = lang_indices[sub]; mask_arr = np.array(mask)
    tlocal = np.array([is_train_row[ci] for ci in mask])
    for qi in idx_t:
        gold = uni_toks(answers_raw[qi])[:CAP]
        if len(gold) < 3: continue
        D, I = idx_map.search(corpus_emb[qi].reshape(1,-1), 31)
        cands = []
        for d, li in zip(D[0], I[0]):
            if li < 0 or not tlocal[int(li)]: continue
            ci = int(mask_arr[int(li)])
            if ci == qi or corpus_q_stripped[ci] == corpus_q_stripped[qi]: continue
            cands.append((ci, float(d), uni_r1(gold, uni_toks(answers_raw[ci])[:CAP])))
        if len(cands) < 4: continue
        best = max(cands, key=lambda c: c[2])
        if best[2] < 0.30: continue
        negs = [c for c in cands if c[1] > best[1]-1e-6 and c[2] < best[2]-0.25][:2]
        for ng in negs:
            triplets.append((f"{PREFIX}{questions_raw[qi]}",
                             f"{PREFIX}{questions_raw[best[0]]}",
                             f"{PREFIX}{questions_raw[ng[0]]}"))
random.shuffle(triplets); triplets = triplets[:16000]
print(f"triplets: {len(triplets)}")

model = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0'); model.max_seq_length = 128
loader = DataLoader([InputExample(texts=list(t)) for t in triplets],
                    shuffle=True, batch_size=12, drop_last=True)
model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
          epochs=1, warmup_steps=int(0.1*len(loader)),
          optimizer_params={'lr': 1e-5}, use_amp=True, show_progress_bar=True)
model.save(str(FT2_DIR))
enc = lambda T: model.encode([f"{PREFIX}{t}" for t in T], batch_size=64,
        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
np.save(CACHE/'ft2_corpus.npy', enc(questions_raw))
np.save(CACHE/'ft2_val.npy',    enc(val_qs))
np.save(CACHE/'ft2_test.npy',   enc(test_qs))
# Deploy ONLY via interpolation: score = beta*AfriE5 + (1-beta)*FT2, beta tuned per language.
'''

# ============================================================================
# D. PER-LANGUAGE RETRIEVER ADAPTER (Lug_Uga)  [ADOPTED, beta=0.8]
#    Answer-similarity supervision: pos = neighbor with answer-R1>=0.5,
#    neg = closer-ranked neighbor with answer-R1<=0.2. One language only.
# ============================================================================
PERLANG_ADAPTER = r'''
import torch, gc, random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
random.seed(42); np.random.seed(42)
AFRIE5_DIR = OUTPUT_DIR / 'afrie5-final-model'; PREFIX = "query: "
TARGETS = ['Lug_Uga']     # Eng_Uga was tried and rejected (supply too thin)
train_q_set = set(q.strip() for q in train_df['input'].dropna().astype(str))
is_train_row = np.array([q in train_q_set for q in corpus_q_stripped])

def mine_lang(sub, cap=6000):
    idx_t = [i for i in range(len(combined)) if subsets_raw[i]==sub and is_train_row[i]]
    random.shuffle(idx_t); idx_t = idx_t[:cap]
    ix, mask = lang_indices[sub]; mask_arr = np.array(mask)
    tmask = np.array([is_train_row[ci] for ci in mask]); trips = []
    for qi in idx_t:
        gold = uni_toks(answers_raw[qi])[:CAP]
        if len(gold) < 3: continue
        D, I = ix.search(corpus_emb[qi].reshape(1,-1), 25)
        cands = []
        for d, li in zip(D[0], I[0]):
            if li < 0 or not tmask[int(li)]: continue
            ci = int(mask_arr[int(li)])
            if ci == qi or corpus_q_stripped[ci] == corpus_q_stripped[qi]: continue
            cands.append((ci, float(d), uni_r1(gold, uni_toks(answers_raw[ci])[:CAP])))
        pos = [c for c in cands if c[2] >= 0.5]
        if not pos: continue
        best = max(pos, key=lambda c: c[2])
        negs = [c for c in cands if c[1] >= best[1]-0.02 and c[2] <= 0.2][:2]
        for ng in negs:
            trips.append((f"{PREFIX}{questions_raw[qi]}",
                          f"{PREFIX}{questions_raw[best[0]]}",
                          f"{PREFIX}{questions_raw[ng[0]]}"))
    return trips

for sub in TARGETS:
    trips = mine_lang(sub); print(f"{sub}: {len(trips)} triplets")
    if len(trips) < 500: print("  too thin, skip"); continue
    import bitsandbytes as bnb
    model = SentenceTransformer(str(AFRIE5_DIR), device='cuda:0'); model.max_seq_length = 96
    model[0].auto_model.gradient_checkpointing_enable()
    loader = DataLoader([InputExample(texts=list(t)) for t in trips],
                        shuffle=True, batch_size=8, drop_last=True)
    model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
              epochs=1, warmup_steps=int(0.1*len(loader)),
              optimizer_class=bnb.optim.AdamW8bit, optimizer_params={'lr': 8e-6},
              use_amp=True, show_progress_bar=True)
    out_dir = OUTPUT_DIR / f'afrie5-{sub.lower()}'; model.save(str(out_dir))
    _, mask = lang_indices[sub]
    enc = lambda T: model.encode([f"{PREFIX}{t}" for t in T], batch_size=64,
            normalize_embeddings=True).astype(np.float32)
    np.save(CACHE/f'pl_{sub}_corpus.npy', enc([questions_raw[ci] for ci in mask]))
    # deploy: score = 0.8*AfriE5 + 0.2*adapter on this language only
'''

# ============================================================================
# E. PSEUDO-LABELING THE CE  [UNTRIED — last experiment, not completed]
#    Run the deployed CE over test, keep confident pairs, retrain. Bounded EV
#    (CE feeds only Aka/Amh). Gate hard vs the deployed CE-stitch with auto-revert.
# ============================================================================
PSEUDO_LABEL_CE = r'''
# Requires Section A reload (ce_scores from ce-reranker-v2) + bootstrap.
import torch, random, numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
random.seed(42); np.random.seed(42)

# 1) regenerate the ORIGINAL pairs (identical mining as Section B) -> orig_pairs
#    [paste the Section B mining block, store result as `orig_pairs`]

# 2) pseudo-label test
POS_THR, NEG_THR = 0.95, 0.05
pseudo_pairs = []
for i in tqdm(range(len(test_df)), desc="pseudo-label"):
    sub = test_subs[i]
    pool = get_same_lang_candidates(test_qs[i].strip(), test_emb[i], sub,
                                    k=K_CANDIDATES, exclude_exact=False)
    if not pool: continue
    cqs = [questions_raw[c['idx']] for c in pool]
    cs = ce_scores(test_qs[i], cqs)
    b = int(np.argmax(cs))
    if cs[b] < POS_THR: continue
    negs = [j for j in np.argsort(cs) if cs[j] < NEG_THR][:2]
    if not negs: continue
    pseudo_pairs.append((test_qs[i], cqs[b], 1))
    for j in negs: pseudo_pairs.append((test_qs[i], cqs[j], 0))

# 3) retrain on orig_pairs + pseudo_pairs (identical config), save ce-reranker-v2-pl
# 4) gate: pl-CE-stitch vs old-CE-stitch on Aka/Amh holdout; adopt iff holdΔ > +0.003
#    (full cells in the session transcript; sanity anchor: old Aka CE-stitch ~0.2288)
'''

# ============================================================================
# F. QWEN2.5-7B LoRA  [epoch 1 ADOPTED for Eng_Gha; epochs 2-3 REJECTED]
#    Reference config (resume cell). Epoch-1 adapter = qwen-ft-health (deployed).
# ============================================================================
QWEN_LORA_CONFIG = r'''
# SFTConfig: num_train_epochs=1 (deployed), per_device_train_batch_size=8,
#   gradient_accumulation_steps=4 (eff batch 32), learning_rate=1e-4 (epoch1)
#   / 5e-5 (resume), lr_scheduler_type='cosine', warmup_steps=30, max_grad_norm=0.3,
#   bf16=True, gradient_checkpointing=True, optim='adamw_torch_fused',
#   max_length=1024, packing=False, completion_only_loss=True, seed=42
# LoRA r=16. Training text in ft_train_data.json. Prompt: "multilingual health expert,
#   use EXACT words from references, answer in {lang}". Deploy generation on Eng_Gha only.
# Full resume cell is in the transcript; epoch-1 adapter on Drive is authoritative.
'''

if __name__ == "__main__":
    print("Cells: A reload, B CE-reranker(adopted), C FT2(interp), "
          "D per-lang adapter(adopted), E pseudo-label(untried), F Qwen config.")
