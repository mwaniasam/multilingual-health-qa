"""
=============================================================================
AfriE5-Large Retrieval — Purpose-built African language embeddings
=============================================================================
Same PROVEN recipe as v4 (0.49 → 0.57 → 0.6459), but with
McGill-NLP/AfriE5-Large-instruct — trained on Twi (Akan), Amharic, Swahili.

Cell 1: Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

Cell 2: Install packages
    !pip install -q sentence-transformers faiss-cpu rouge-score tqdm scikit-learn

Cell 3: Paste this entire script
=============================================================================
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
import faiss
import gc
import traceback
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from datetime import datetime
from collections import defaultdict

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# MOUNT GOOGLE DRIVE + SET PATHS
# ============================================================
try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
    else:
        log("Google Drive already mounted.")
    DATA_DIR = Path('/content/drive/MyDrive/multilingual-health-qa/data')
    OUTPUT_DIR = Path('/content/drive/MyDrive/multilingual-health-qa/outputs')
    log(f"Data: {DATA_DIR}")
except ImportError:
    # Not on Colab — use local paths
    DATA_DIR = Path('/kaggle/input/datasets/samuelmwania1/multilingual-health-qa-data/')
    if not DATA_DIR.exists():
        import glob
        found = glob.glob('/kaggle/input/**/Train.csv', recursive=True)
        DATA_DIR = Path(found[0]).parent if found else Path('data/raw/')
    OUTPUT_DIR = Path('/kaggle/working/')
    log(f"Data: {DATA_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log(f"Output: {OUTPUT_DIR}")

# ============================================================
# LOAD & VALIDATE DATA
# ============================================================
log("Loading data...")
train_df = pd.read_csv(DATA_DIR / 'Train.csv')
val_df = pd.read_csv(DATA_DIR / 'Val.csv')
test_df = pd.read_csv(DATA_DIR / 'Test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

for name, df, req_cols in [
    ('Train', train_df, ['input', 'output', 'subset']),
    ('Val', val_df, ['input', 'output', 'subset']),
    ('Test', test_df, ['ID', 'input', 'subset']),
    ('SampleSub', sample_sub, ['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']),
]:
    missing = [c for c in req_cols if c not in df.columns]
    assert not missing, f"❌ {name}.csv missing columns: {missing}. Has: {list(df.columns)}"
    log(f"  ✅ {name}: {len(df)} rows")

combined = pd.concat([train_df, val_df], ignore_index=True).dropna(subset=['input', 'output'])
combined = combined.reset_index(drop=True)
log(f"Combined: {len(combined)} samples")

questions_raw = combined['input'].fillna('').astype(str).tolist()
answers_raw = combined['output'].fillna('').astype(str).tolist()
subsets_raw = combined['subset'].tolist()

assert len(questions_raw) == len(answers_raw) == len(combined)
log(f"  Sanity check passed: {len(questions_raw)} Q&A pairs")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ============================================================
# GPU INFO
# ============================================================
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu_name} | {gpu_mem:.1f} GB")
else:
    log("⚠️ No GPU!")

# ============================================================
# MODEL CONFIG — AfriE5-Large-instruct
# ============================================================
# AfriE5 uses "Instruct: ...\nQuery: " prefix for queries
# For retrieval, we use "query: " which is the E5 convention
# AfriE5 is built on E5-large-instruct and works with both formats
MODEL_NAME = "McGill-NLP/AfriE5-Large-instruct"
QUERY_PREFIX = "query: "  # E5 convention, works with AfriE5 too

log(f"\n{'='*70}")
log(f"AfriE5-Large Retrieval — African Language Embeddings")
log(f"{'='*70}")
log(f"Model: {MODEL_NAME}")

# ============================================================
# HELPER FUNCTIONS (same as v4 — proven recipe)
# ============================================================
def encode_corpus(mdl, bs=64):
    """Encode all questions in the combined corpus."""
    emb = mdl.encode([f"{QUERY_PREFIX}{q}" for q in questions_raw],
                     batch_size=bs, show_progress_bar=True,
                     normalize_embeddings=True).astype(np.float32)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    log(f"  Encoded corpus: {emb.shape}")
    return emb, idx


def evaluate_val(mdl, fidx, label="", top_k=10):
    """Evaluate retrieval on val set. Returns (rouge1, rougeL)."""
    val_qs = [f"{QUERY_PREFIX}{str(q)}" for q in val_df['input'].fillna('').tolist()]
    val_emb = mdl.encode(val_qs, batch_size=64, show_progress_bar=True,
                         normalize_embeddings=True).astype(np.float32)
    r1s, rls = [], []
    for i in range(len(val_df)):
        q = str(val_df.iloc[i]['input']).strip()
        ref = str(val_df.iloc[i]['output']).strip()
        if not ref:
            continue
        D, I = fidx.search(val_emb[i:i+1], top_k)
        pred = ''
        for j in range(top_k):
            cand_idx = int(I[0][j])
            if cand_idx >= len(combined):
                continue
            if str(combined.iloc[cand_idx]['input']).strip() != q:
                pred = str(combined.iloc[cand_idx]['output'])
                break
        if not pred:
            pred = str(combined.iloc[int(I[0][0])]['output'])
        r = scorer.score(ref, pred)
        r1s.append(r['rouge1'].fmeasure)
        rls.append(r['rougeL'].fmeasure)
    r1, rl = np.mean(r1s), np.mean(rls)
    log(f"[{label}] Val ROUGE-1: {r1:.4f} | ROUGE-L: {rl:.4f}")
    return r1, rl


def save_submission(mdl, fidx, fname, comment):
    """Generate and save a submission CSV."""
    test_qs = [f"{QUERY_PREFIX}{str(q)}" for q in test_df['input'].fillna('').tolist()]
    test_emb = mdl.encode(test_qs, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True).astype(np.float32)
    rows = []
    for i in range(len(test_df)):
        D, I = fidx.search(test_emb[i:i+1], 10)
        q = str(test_df.iloc[i]['input']).strip()
        answer = ''
        for j in range(10):
            cand_idx = int(I[0][j])
            if cand_idx >= len(combined):
                continue
            cand = str(combined.iloc[cand_idx]['input']).strip()
            if cand != q:
                answer = str(combined.iloc[cand_idx]['output'])
                break
        if not answer:
            answer = str(combined.iloc[int(I[0][0])]['output'])
        rows.append({
            'ID': test_df.iloc[i]['ID'],
            'TargetRLF1': answer, 'TargetR1F1': answer, 'TargetLLM': answer,
        })
    sub = pd.DataFrame(rows)
    sub = sub[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    assert len(sub) == len(sample_sub), f"Row count mismatch: {len(sub)} vs {len(sample_sub)}"
    assert sub['TargetRLF1'].isna().sum() == 0, "Found NaN in TargetRLF1!"
    sub.to_csv(OUTPUT_DIR / fname, index=False)
    log(f"✅ Saved: {fname} | {comment}")


def mine_hn(corpus_emb, fidx, max_neg=2, range_min=0, range_max=30):
    """Mine hard negatives."""
    examples = []
    skipped = 0
    for i in tqdm(range(len(combined)), desc="Mining HN"):
        q, a = questions_raw[i], answers_raw[i]
        if not q.strip() or not a.strip():
            skipped += 1
            continue
        D, I = fidx.search(corpus_emb[i:i+1], range_max)
        negs = []
        for j in range(range_min, range_max):
            idx_j = int(I[0][j])
            if idx_j == i:
                continue
            if idx_j >= len(answers_raw):
                continue
            if answers_raw[idx_j].strip() != a.strip() and len(negs) < max_neg:
                negs.append(f"passage: {answers_raw[idx_j]}")
        texts = [f"{QUERY_PREFIX}{q}", f"passage: {a}"] + negs
        examples.append(InputExample(texts=texts))
    log(f"  Mined {len(examples)} examples ({skipped} skipped)")
    return examples


def build_qq_pairs(corpus_emb, fidx, max_pairs_per_answer=5):
    """Build question-to-question pairs from same-answer groups."""
    answer_groups = defaultdict(list)
    for i, a in enumerate(answers_raw):
        key = a.strip()[:200]
        if key:
            answer_groups[key].append(i)

    qq_examples = []
    for key, indices in tqdm(answer_groups.items(), desc="Q-Q pairs"):
        if len(indices) < 2:
            continue
        pairs_made = 0
        for i in range(len(indices)):
            if pairs_made >= max_pairs_per_answer:
                break
            ai = indices[i]
            pi = indices[(i + 1) % len(indices)]
            if questions_raw[ai].strip() == questions_raw[pi].strip():
                continue
            D, I = fidx.search(corpus_emb[ai:ai+1], 20)
            neg_q = None
            for j in range(20):
                ni = int(I[0][j])
                if ni == ai or ni == pi:
                    continue
                if ni >= len(answers_raw):
                    continue
                if answers_raw[ni].strip()[:200] != key:
                    neg_q = questions_raw[ni]
                    break
            if neg_q:
                qq_examples.append(InputExample(
                    texts=[f"{QUERY_PREFIX}{questions_raw[ai]}", f"{QUERY_PREFIX}{questions_raw[pi]}", f"{QUERY_PREFIX}{neg_q}"]
                ))
                pairs_made += 1

    log(f"Q-Q pairs: {len(qq_examples)}")
    return qq_examples


def do_training_round(mdl, examples, label, epochs, warmup, use_cached):
    """Run one training round with error handling."""
    log(f"  Training {label}: {len(examples)} examples, {epochs} epochs")

    if use_cached:
        try:
            from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
            train_loss = CachedMultipleNegativesRankingLoss(mdl, mini_batch_size=8)
            loader = DataLoader(examples, shuffle=True, batch_size=64)
            log(f"  Using CachedMNRL: effective_batch=64, mini_batch=8")
        except Exception as e:
            log(f"  ⚠️ CachedMNRL failed ({e}), falling back to standard MNRL")
            train_loss = losses.MultipleNegativesRankingLoss(mdl)
            loader = DataLoader(examples, shuffle=True, batch_size=8)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(mdl)
        loader = DataLoader(examples, shuffle=True, batch_size=8)

    steps = len(loader) * epochs
    log(f"  Steps: {steps}")

    try:
        mdl.fit(
            train_objectives=[(loader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup,
            show_progress_bar=True,
            output_path=str(OUTPUT_DIR / f'afrie5-{label}'),
            use_amp=True,
        )
        log(f"  ✅ {label} complete!")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log(f"  ⚠️ OOM! Retrying with smaller batch...")
            gc.collect(); torch.cuda.empty_cache()
            train_loss = losses.MultipleNegativesRankingLoss(mdl)
            loader = DataLoader(examples, shuffle=True, batch_size=4)
            mdl.fit(
                train_objectives=[(loader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup,
                show_progress_bar=True,
                output_path=str(OUTPUT_DIR / f'afrie5-{label}-retry'),
                use_amp=True,
            )
            log(f"  ✅ {label} complete (with smaller batch)!")
        else:
            raise

    del train_loss, loader
    gc.collect(); torch.cuda.empty_cache()


# ============================================================
# MAIN PIPELINE
# ============================================================
model = SentenceTransformer(MODEL_NAME, device='cuda:0' if torch.cuda.is_available() else 'cpu')
log(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

best_r1 = 0
best_rl = 0
best_file = ""
all_results = []

# Check if CachedMNRL is available
try:
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    USE_CACHED = True
    log("✅ CachedMNRL available! Will use effective_batch=64")
except ImportError:
    USE_CACHED = False
    log("⚠️ CachedMNRL not available, using standard MNRL")

# --- BASELINE (before any training) ---
log("\n--- BASELINE (no fine-tuning) ---")
corpus_emb, fidx = encode_corpus(model)
r1_base, rl_base = evaluate_val(model, fidx, "BASELINE")
save_submission(model, fidx, "afrie5_baseline.csv", f"No FT. Val R1={r1_base:.4f} RL={rl_base:.4f}")
all_results.append(("baseline", r1_base, rl_base))

# ============================================================
# PHASE 1a: Round 1 HN mining + CachedMNRL
# ============================================================
try:
    log("\n" + "=" * 50)
    log("PHASE 1a: Round 1 — HN mining + CachedMNRL large batch")
    log("=" * 50)

    examples = mine_hn(corpus_emb, fidx, max_neg=2, range_min=0, range_max=30)
    del corpus_emb; gc.collect(); torch.cuda.empty_cache()

    do_training_round(model, examples, "hn-r1", epochs=3, warmup=200, use_cached=USE_CACHED)
    del examples; gc.collect(); torch.cuda.empty_cache()

    corpus_emb, fidx = encode_corpus(model)
    r1, rl = evaluate_val(model, fidx, "HN-R1")
    save_submission(model, fidx, "afrie5_hn_r1.csv", f"CachedMNRL R1. Val R1={r1:.4f} RL={rl:.4f}")
    all_results.append(("hn_r1", r1, rl))
    if r1 > best_r1:
        best_r1, best_rl, best_file = r1, rl, "afrie5_hn_r1.csv"

except Exception as e:
    log(f"❌ PHASE 1a FAILED: {e}")
    traceback.print_exc()
    log("Continuing to next phase...")
    corpus_emb, fidx = encode_corpus(model)

# ============================================================
# PHASE 1b: Round 2 HN mining (harder negatives)
# ============================================================
try:
    log("\n" + "=" * 50)
    log("PHASE 1b: Round 2 — Harder negatives, range_min=2")
    log("=" * 50)

    examples = mine_hn(corpus_emb, fidx, max_neg=3, range_min=2, range_max=25)
    del corpus_emb; gc.collect(); torch.cuda.empty_cache()

    do_training_round(model, examples, "hn-r2", epochs=2, warmup=100, use_cached=USE_CACHED)
    del examples; gc.collect(); torch.cuda.empty_cache()

    corpus_emb, fidx = encode_corpus(model)
    r1, rl = evaluate_val(model, fidx, "HN-R2")
    save_submission(model, fidx, "afrie5_hn_r2.csv", f"CachedMNRL R2. Val R1={r1:.4f} RL={rl:.4f}")
    all_results.append(("hn_r2", r1, rl))
    if r1 > best_r1:
        best_r1, best_rl, best_file = r1, rl, "afrie5_hn_r2.csv"

except Exception as e:
    log(f"❌ PHASE 1b FAILED: {e}")
    traceback.print_exc()
    log("Continuing to next phase...")
    corpus_emb, fidx = encode_corpus(model)

# ============================================================
# PHASE 2: Q-Q Training
# ============================================================
try:
    log("\n" + "=" * 50)
    log("PHASE 2: Q-Q Training")
    log("=" * 50)

    qq_examples = build_qq_pairs(corpus_emb, fidx, max_pairs_per_answer=5)
    del corpus_emb; gc.collect(); torch.cuda.empty_cache()

    if len(qq_examples) > 0:
        do_training_round(model, qq_examples, "qq", epochs=2, warmup=100, use_cached=USE_CACHED)
        del qq_examples; gc.collect(); torch.cuda.empty_cache()

        corpus_emb, fidx = encode_corpus(model)
        r1, rl = evaluate_val(model, fidx, "HN+QQ")
        save_submission(model, fidx, "afrie5_hn_qq.csv", f"CachedMNRL+QQ. Val R1={r1:.4f} RL={rl:.4f}")
        all_results.append(("hn_qq", r1, rl))
        if r1 > best_r1:
            best_r1, best_rl, best_file = r1, rl, "afrie5_hn_qq.csv"
    else:
        log("⚠️ No Q-Q pairs generated, skipping")
        corpus_emb, fidx = encode_corpus(model)

except Exception as e:
    log(f"❌ PHASE 2 FAILED: {e}")
    traceback.print_exc()
    log("Continuing...")
    corpus_emb, fidx = encode_corpus(model)

# Save model to Drive
try:
    model.save(str(OUTPUT_DIR / 'afrie5-final-model'))
    log("Model saved to Google Drive!")
except:
    pass

# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n" + "=" * 70)
log("🏆 AfriE5-Large RETRIEVAL COMPLETE")
log("=" * 70)
log("")
log("ALL RESULTS:")
log(f"{'Experiment':<20} {'ROUGE-1':>8} {'ROUGE-L':>8}")
log("-" * 40)
for name, r1_val, rl_val in all_results:
    marker = " ← BEST" if name in best_file else ""
    log(f"{name:<20} {r1_val:>8.4f} {rl_val:>8.4f}{marker}")
log("-" * 40)
log("")
log(f"Previous best (E5-base FT): LB=0.6459")
log(f"🏆 BEST THIS RUN: {best_file} (Val R1={best_r1:.4f})")
log(f"📥 FILES TO SUBMIT (in Google Drive → multilingual-health-qa/outputs/):")
for f in sorted((OUTPUT_DIR).glob("afrie5_*.csv")):
    log(f"  → {f.name}")
log("")
log("SUBMIT PRIORITY:")
log("  1. Best val score file (shown above)")
log("  2. afrie5_hn_r2.csv (safest — 2 HN rounds)")
log("  3. afrie5_hn_qq.csv (if Q-Q helped)")
