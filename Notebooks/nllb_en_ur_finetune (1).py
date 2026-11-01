# %% [markdown]
# # English -> Urdu fine-tuning of NLLB
# Datasets: english_urdu.csv, parallel-corpus.csv, quran.csv
# Pipeline: load -> auto-detect columns -> clean -> filter -> dedupe -> split
#           -> zero-shot baseline -> fine-tune -> evaluate -> error analysis -> save

# %% 0. Install (Kaggle / Colab). Uncomment what you need.
# !pip install -q -U transformers datasets sacrebleu sentencepiece accelerate
# Optional extras:
# !pip install -q sentence-transformers unbabel-comet

# %% 1. Imports and configuration
import os, re, json, html, inspect, unicodedata, warnings
from collections import Counter

# reduce GPU memory fragmentation (must be set BEFORE torch touches the GPU)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sacrebleu.metrics import BLEU, CHRF
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed,
)

warnings.filterwarnings("ignore")

# ---- files: name -> domain tag (used for stratified splits and per-domain scores)
DATASET_DIRS = {   # domain -> Kaggle folder
    "general":  "/kaggle/input/datasets/atifkhan12/english-urdu",
    "parallel": "/kaggle/input/datasets/atifkhan12/parallel-corpus",
    "quran":    "/kaggle/input/datasets/atifkhan12/quran-df",
}
EXPECTED_NAMES = {"general": "english_urdu", "parallel": "parallel-corpus", "quran": "quran"}


def _pick_csv(folder, expected):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder} not found. Add this dataset to the notebook (Add Input).")
    cands = [os.path.join(dp, fn) for dp, _, fns in os.walk(folder) for fn in fns
             if fn.lower().endswith((".csv", ".tsv", ".txt"))]
    if not cands:
        raise FileNotFoundError(f"No CSV file inside {folder}")
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    close = [c for c in cands if norm(expected) in norm(os.path.basename(c))]
    return max(close or cands, key=os.path.getsize)     # best name match, else the largest file


FILES, DATA_PATHS, SEARCH_DIRS = {}, {}, []   # same variables the rest of the script uses
for _domain, _folder in DATASET_DIRS.items():
    _path = _pick_csv(_folder, EXPECTED_NAMES[_domain])
    _name = os.path.basename(_path)
    if _name in FILES:
        raise ValueError(f"Two datasets contain a file called {_name}; rename one or set FILES manually.")
    FILES[_name] = _domain
    DATA_PATHS[_name] = _path
    SEARCH_DIRS.append(os.path.dirname(_path))
    print(f"{_domain:9s} -> {_path}")

# If auto-detection picks the wrong columns, set them here:
# COLUMN_OVERRIDES = {"quran.csv": ("English_Column_Name", "Urdu_Column_Name")}
COLUMN_OVERRIDES = {}

# ---- model / languages
MODEL_NAME = "facebook/nllb-200-distilled-600M"   # try ...-1.3B if you have the GPU memory
SRC_LANG, TGT_LANG = "eng_Latn", "urd_Arab"

# ---- preprocessing
MAX_LEN = 200                 # max tokens per side; longer pairs are DROPPED (not truncated)
MAX_WORDS = 200
MIN_SCRIPT_RATIO = 0.6        # share of letters that must be in the expected script
WORD_RATIO_RANGE = (1 / 3, 3.0)
STRIP_URDU_DIACRITICS = True
USE_LABSE_FILTER = False      # needs sentence-transformers + internet to download LaBSE
LABSE_THRESHOLD = 0.60

# ---- splits
VAL_MAX, TEST_MAX = 3000, 3000
VAL_EVAL_SUBSET = 800         # validation rows used for per-epoch eval during training (speed)
MAX_TRAIN_ROWS = None         # e.g. 50000 for a quick smoke test

# ---- training
SEED = 42
EPOCHS = 3
LR = 1e-4                     # with Adafactor; use ~3e-5 to 5e-5 if you switch to AdamW
TRAIN_BS, EVAL_BS, GRAD_ACC = 4, 8, 4           # per GPU; effective batch = 4 x n_gpus x 4 (32 on 2 GPUs)
                                                # CUDA out of memory? -> TRAIN_BS=2, GRAD_ACC=8
FREEZE_EMBEDDINGS = True      # saves lots of memory (256k-vocab embeddings are tied to lm_head)
OPTIM = "adafactor"           # "adafactor" (low memory) or "adamw_torch"
LABEL_SMOOTHING = 0.0         # 0.1 builds a full log-softmax over the 256k vocab and often runs out of memory on T4s
RESUME_FROM = None            # path to a checkpoint to resume

# ---- evaluation
RUN_BASELINE = True           # zero-shot NLLB scored on the same test set
NUM_BEAMS = 5
GEN_BATCH = 16               # lower to 8 if generation runs out of memory
RUN_COMET = False             # needs unbabel-comet + model download

OUT_DIR = "/kaggle/working/nllb_en_ur" if os.path.isdir("/kaggle/working") else "./nllb_en_ur"
os.makedirs(OUT_DIR, exist_ok=True)

set_seed(SEED)
USE_CUDA = torch.cuda.is_available()
print("CUDA:", USE_CUDA, "|", torch.cuda.get_device_name(0) if USE_CUDA else "CPU (training will be very slow)")

# %% 2. Text cleaning helpers
_ZW = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_FOOT = re.compile(r"\[\d+\]")
_WS = re.compile(r"\s+")
# harakat, superscript alef, Quranic annotation marks (U+06D6-U+06ED), tatweel
_DIAC = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")

EN_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                          "\u2013": "-", "\u2014": "-", "\u00a0": " "})

# Arabic-variant letters -> Urdu letters; Arabic-Indic digits -> ASCII
_ur_map = {"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۀ": "ۂ"}
_ur_map.update({chr(0x0660 + i): str(i) for i in range(10)})
_ur_map.update({chr(0x06F0 + i): str(i) for i in range(10)})
UR_MAP = str.maketrans(_ur_map)


def clean_en(t):
    t = html.unescape(str(t))
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = unicodedata.normalize("NFKC", t).translate(EN_PUNCT)
    t = _ZW.sub("", t)
    t = _CTRL.sub(" ", t)
    t = _FOOT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def clean_ur(t):
    t = html.unescape(str(t))
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)
    t = _ZW.sub("", t)
    t = _CTRL.sub(" ", t)
    t = _FOOT.sub(" ", t)
    if STRIP_URDU_DIACRITICS:
        t = _DIAC.sub("", t)
    t = t.translate(UR_MAP)
    return _WS.sub(" ", t).strip()


def script_ratio(t, kind):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    if kind == "ur":
        n = sum(("\u0600" <= c <= "\u06ff") or ("\u0750" <= c <= "\u077f") or ("\ufb50" <= c <= "\ufeff")
                for c in letters)
    else:
        n = sum(c.isascii() or ("\u00c0" <= c <= "\u024f") for c in letters)
    return n / len(letters)


# %% 3. Robust CSV reading and automatic column detection
def find_file(name):
    """Resolve the CSV for a dataset: exact file name first, else the only/best-matching CSV in the folder."""
    p = DATA_PATHS[name]
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{p} does not exist. Check the dataset is added to this notebook.")
    csvs = []
    for dp, _, fns in os.walk(p):
        for fn in fns:
            if fn.lower().endswith((".csv", ".tsv", ".txt")):
                csvs.append(os.path.join(dp, fn))
    print(f"  files in {p}: {[os.path.basename(c) for c in csvs]}")
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {p}")
    exact = [c for c in csvs if os.path.basename(c).lower() == name.lower()]
    if exact:
        return exact[0]
    if len(csvs) == 1:
        return csvs[0]
    stem = os.path.splitext(name)[0].lower().replace("-", "").replace("_", "")
    close = [c for c in csvs if stem in os.path.basename(c).lower().replace("-", "").replace("_", "")]
    if close:
        return close[0]
    raise FileNotFoundError(f"Several CSVs in {p} and none match '{name}': set DATA_PATHS['{name}'] to the exact file.")


def read_csv_robust(path):
    kw = dict(dtype=str, keep_default_na=False, on_bad_lines="skip")
    last, df, used = None, None, None
    for enc in ["utf-8-sig", "utf-8", "utf-16", "cp1256", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc, **kw)
            used = enc
            break
        except Exception as e:  # noqa
            last = e
    if df is None:
        raise last
    # headerless file? (a "header" that looks like data)
    if any(len(str(c)) > 60 or re.search(r"[\u0600-\u06ff]", str(c)) for c in df.columns):
        df = pd.read_csv(path, encoding=used, header=None, **kw)
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df


UR_WORDS = {"کے", "کی", "ہے", "اور", "میں", "کو", "سے", "کہ", "ہیں", "نے", "کا", "پر"}
EN_WORDS = {"the", "of", "and", "to", "in", "is", "a", "that", "for", "with", "he", "you", "was", "it"}


def _urdu_score(s):
    s = s.astype(str).head(500)
    if not len(s):
        return 0.0
    return float(s.apply(lambda x: any(w in UR_WORDS for w in x.split())).mean())


def _english_score(s):
    s = s.astype(str).head(500)
    if not len(s):
        return 0.0
    hit = s.apply(lambda x: any(w in EN_WORDS for w in x.lower().split())).mean()
    asc = s.apply(lambda x: script_ratio(x, "en")).mean()
    return float(hit * asc)


def detect_columns(df, fname):
    if fname in COLUMN_OVERRIDES:
        return COLUMN_OVERRIDES[fname]
    en = {c: _english_score(df[c]) for c in df.columns}
    en_col = max(en, key=en.get)
    ur = {c: _urdu_score(df[c]) for c in df.columns if c != en_col}
    ur_col = max(ur, key=ur.get)
    print(f"  columns: {list(df.columns)}")
    print(f"  english scores: { {k: round(v, 2) for k, v in en.items()} }")
    print(f"  urdu scores:    { {k: round(v, 2) for k, v in ur.items()} }")
    print(f"  -> English = '{en_col}', Urdu = '{ur_col}'")
    if en[en_col] < 0.2 or ur[ur_col] < 0.2:
        raise ValueError(f"Could not confidently detect columns in {fname}. "
                         f"Set COLUMN_OVERRIDES['{fname}'] = (english_col, urdu_col).")
    return en_col, ur_col


# %% 4. Load + clean + filter each dataset
def load_and_clean(fname, domain):
    print(f"\n=== {fname} ({domain}) ===")
    path = find_file(fname)
    raw = read_csv_robust(path)
    en_col, ur_col = detect_columns(raw, fname)
    df = pd.DataFrame({"src": raw[en_col], "tgt": raw[ur_col]})
    log = {"raw": len(df)}

    df["src"] = df["src"].map(clean_en)
    df["tgt"] = df["tgt"].map(clean_ur)

    df = df[(df.src.str.len() >= 2) & (df.tgt.str.len() >= 2)]
    log["non_empty"] = len(df)

    df = df[df.src.str.lower() != df.tgt.str.lower()]
    log["src!=tgt"] = len(df)

    df = df[df.src.map(lambda x: script_ratio(x, "en")) >= MIN_SCRIPT_RATIO]
    df = df[df.tgt.map(lambda x: script_ratio(x, "ur")) >= MIN_SCRIPT_RATIO]
    log["script_ok"] = len(df)

    ns, nt = df.src.str.split().str.len(), df.tgt.str.split().str.len()
    keep = (ns <= MAX_WORDS) & (nt <= MAX_WORDS)
    r = ns / nt.clip(lower=1)
    bad_ratio = (np.minimum(ns, nt) >= 5) & ((r > WORD_RATIO_RANGE[1]) | (r < WORD_RATIO_RANGE[0]))
    df = df[keep & ~bad_ratio]
    log["len_ratio_ok"] = len(df)

    df = df.drop_duplicates(["src", "tgt"])
    log["dedup_pairs"] = len(df)

    df["domain"] = domain
    print("  filter log:", log)
    return df.reset_index(drop=True)


frames = [load_and_clean(f, d) for f, d in FILES.items()]
data = pd.concat(frames, ignore_index=True)
print("\nCombined before cross-dataset dedup:", len(data))

# %% 5. Cross-dataset dedup (prevents the same source sentence landing in train AND test)
_key = lambda s: re.sub(r"\W+", "", s.lower())
data["_k"] = data.src.map(_key)
data = data.drop_duplicates("_k", keep="first").drop(columns="_k").reset_index(drop=True)
print("After source-level dedup:", len(data))

# %% 6. Optional: LaBSE similarity filter (removes misaligned pairs)
def labse_filter(df, thr, bs=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/LaBSE", device="cuda" if USE_CUDA else "cpu")
    a = m.encode(df.src.tolist(), batch_size=bs, normalize_embeddings=True, show_progress_bar=True)
    b = m.encode(df.tgt.tolist(), batch_size=bs, normalize_embeddings=True, show_progress_bar=True)
    sim = (a * b).sum(1)
    print(f"LaBSE: dropping {(sim < thr).sum()} / {len(df)} pairs below {thr}")
    out = df[sim >= thr].reset_index(drop=True)
    del m
    torch.cuda.empty_cache()
    return out


if USE_LABSE_FILTER:
    data = labse_filter(data, LABSE_THRESHOLD)

# %% 7. Tokenizer + token-length filter (+ <unk> sanity check)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
TGT_ID = tok.convert_tokens_to_ids(TGT_LANG)


def token_lengths(texts, target=False, chunk=50_000):
    lens, unk = [], 0
    for i in range(0, len(texts), chunk):
        part = texts[i:i + chunk]
        ids = tok(text_target=part, truncation=False)["input_ids"] if target else tok(part, truncation=False)["input_ids"]
        lens.extend(len(x) for x in ids)
        if target:
            unk += sum(x.count(tok.unk_token_id) for x in ids)
    return np.array(lens), unk


ls, _ = token_lengths(data.src.tolist())
lt, unk = token_lengths(data.tgt.tolist(), target=True)
print(f"Token length  src: mean={ls.mean():.1f} p95={np.percentile(ls, 95):.0f} | "
      f"tgt: mean={lt.mean():.1f} p95={np.percentile(lt, 95):.0f}")
print(f"<unk> rate on Urdu targets: {unk / lt.sum():.5%}")
data = data[(ls <= MAX_LEN) & (lt <= MAX_LEN)].reset_index(drop=True)
print("After token-length filter:", len(data))
print(data.domain.value_counts().to_string())

# %% 8. Stratified train / val / test split (stratified by dataset)
def strat_split(df, test_size):
    try:
        return train_test_split(df, test_size=test_size, stratify=df.domain, random_state=SEED)
    except ValueError:
        return train_test_split(df, test_size=test_size, random_state=SEED)


n_test = min(TEST_MAX, int(len(data) * 0.05))
n_val = min(VAL_MAX, int(len(data) * 0.05))
rest, test = strat_split(data, n_test)
train, val = strat_split(rest, n_val)
if MAX_TRAIN_ROWS:
    train = train.sample(min(MAX_TRAIN_ROWS, len(train)), random_state=SEED)
train, val, test = [d.reset_index(drop=True) for d in (train, val, test)]
print(f"train={len(train)} val={len(val)} test={len(test)}")

train.to_csv(f"{OUT_DIR}/train_clean.csv", index=False)
val.to_csv(f"{OUT_DIR}/val_clean.csv", index=False)
test.to_csv(f"{OUT_DIR}/test_clean.csv", index=False)

# %% 9. Model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.generation_config.forced_bos_token_id = TGT_ID   # generation settings live ONLY in generation_config
if USE_CUDA:
    model.to("cuda")
if FREEZE_EMBEDDINGS:
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_train / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

# %% 10. Translation + metric helpers
@torch.no_grad()
def translate(texts, batch_size=GEN_BATCH, num_beams=NUM_BEAMS, max_new_tokens=MAX_LEN + 20):
    model.eval()
    order = np.argsort([-len(t) for t in texts])     # longest first: fewer padding tokens, OOM shows up early
    out = [None] * len(texts)
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        batch = [texts[j] for j in idx]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_CUDA):
            gen = model.generate(**enc, forced_bos_token_id=TGT_ID, num_beams=num_beams,
                                 max_new_tokens=max_new_tokens)
        for j, d in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = d
    return out


_chrf = CHRF(word_order=2)   # chrF++
_bleu = BLEU()


def corpus_scores(h, r):
    return {"BLEU": _bleu.corpus_score(h, [r]).score, "chrF++": _chrf.corpus_score(h, [r]).score}


def evaluate_system(name, hyps, refs, domains):
    h = [clean_ur(x) for x in hyps]      # same normalisation on both sides, so letter variants don't penalise
    r = [clean_ur(x) for x in refs]
    rows = [{"system": name, "subset": "ALL", "n": len(h), **corpus_scores(h, r)}]
    for d in sorted(set(domains)):
        idx = [i for i, x in enumerate(domains) if x == d]
        rows.append({"system": name, "subset": d, "n": len(idx),
                     **corpus_scores([h[i] for i in idx], [r[i] for i in idx])})
    return pd.DataFrame(rows)


def sentence_chrf(hyps, refs):
    return np.array([_chrf.sentence_score(clean_ur(h), [clean_ur(r)]).score for h, r in zip(hyps, refs)])


def paired_bootstrap(a, b, n=1000, seed=0):
    """Is system B better than A? Uses sentence-level chrF++ scores."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = (b[idx] - a[idx]).mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), (lo, hi), float((diffs <= 0).mean())


def has_repetition(t, n=3, k=3):
    w = t.split()
    if len(w) < n * k:
        return False
    return max(Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)).values()) >= k


def error_analysis(df, col, out_path):
    h = df[col].fillna("").tolist()
    ref_len = df.tgt.str.split().str.len().clip(lower=1)
    hyp_len = pd.Series([len(x.split()) for x in h])
    stats = {
        "empty_outputs": int(sum(1 for x in h if not x.strip())),
        "hyp/ref length ratio (mean)": float((hyp_len / ref_len).mean()),
        "outputs w/ >20% Latin letters (untranslated)": int(sum(script_ratio(x, "en") > 0.2 for x in h if x.strip())),
        "outputs w/ repeated 3-grams (degeneration)": int(sum(has_repetition(x) for x in h)),
    }
    print(f"\nError analysis [{col}]:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    # length-bucket scores
    buckets = pd.cut(df.src.str.split().str.len(), [0, 5, 10, 20, 40, 1000], labels=["1-5", "6-10", "11-20", "21-40", "40+"])
    print("  sentence chrF++ by source length:")
    print(df.groupby(buckets)[f"chrf_{col}"].mean().round(1).to_string())
    df.sort_values(f"chrf_{col}").head(50)[["src", "tgt", col, f"chrf_{col}", "domain"]].to_csv(out_path, index=False)
    print(f"  50 worst examples -> {out_path}")


# %% 11. Zero-shot baseline on the test set (so you know whether fine-tuning really helps)
results = []
test_out = test.copy()
if RUN_BASELINE:
    print("\nTranslating test set with zero-shot NLLB ...")
    test_out["baseline"] = translate(test.src.tolist())
    results.append(evaluate_system("zero-shot NLLB", test_out.baseline.tolist(), test.tgt.tolist(), test.domain.tolist()))
    print(results[-1].round(2).to_string(index=False))

# %% 12. Tokenise datasets, collator, Trainer
def tokenize_batch(b):
    return tok(b["src"], text_target=b["tgt"], max_length=MAX_LEN, truncation=True)


def to_hf(df):
    ds = Dataset.from_pandas(df[["src", "tgt"]].reset_index(drop=True))
    return ds.map(tokenize_batch, batched=True, remove_columns=["src", "tgt"], desc="tokenising")


train_ds = to_hf(train)
val_ds = to_hf(val.sample(min(len(val), VAL_EVAL_SUBSET), random_state=SEED))
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right


class NLLBCollator(DataCollatorForSeq2Seq):
    """NLLB (M2M100) has no prepare_decoder_input_ids_from_labels(), and label smoothing removes `labels`
    before the forward pass, so the decoder would receive no inputs. Build decoder_input_ids ourselves."""

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], tok.pad_token_id, model.config.decoder_start_token_id)
        return batch


collator = NLLBCollator(tok, model=model, pad_to_multiple_of=8, label_pad_token_id=-100)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tok.pad_token_id)
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    h = [clean_ur(x) for x in tok.batch_decode(preds, skip_special_tokens=True)]
    r = [clean_ur(x) for x in tok.batch_decode(labels, skip_special_tokens=True)]
    s = corpus_scores(h, r)
    return {"chrf": s["chrF++"], "bleu": s["BLEU"], "gen_len": float(np.mean([len(x.split()) for x in h]))}


ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
steps_per_epoch = -(-len(train_ds) // (TRAIN_BS * GRAD_ACC))
warmup_steps = max(1, int(0.05 * steps_per_epoch * EPOCHS))      # 5% warmup, as an int (works in every version)
print(f"steps/epoch={steps_per_epoch}, total={steps_per_epoch * EPOCHS}, warmup={warmup_steps}")
args_kw = dict(
    output_dir=f"{OUT_DIR}/checkpoints",
    per_device_train_batch_size=TRAIN_BS, per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR, num_train_epochs=EPOCHS, warmup_steps=warmup_steps, weight_decay=0.01,
    lr_scheduler_type="linear", label_smoothing_factor=LABEL_SMOOTHING,
    optim=OPTIM, fp16=USE_CUDA,
    group_by_length=True,                          # older transformers
    train_sampling_strategy="group_by_length",     # newer transformers (unsupported keys are dropped below)
    predict_with_generate=True, generation_max_length=MAX_LEN, generation_num_beams=1,
    save_strategy="epoch", logging_steps=100,
    load_best_model_at_end=True, metric_for_best_model="chrf", greater_is_better=True,
    save_total_limit=2, report_to="none", seed=SEED, dataloader_num_workers=2,
)
args_kw["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"
_ignored = [k for k in args_kw if k not in ta_params]
if _ignored:
    print("Ignored (not supported by this transformers version):", _ignored)
args = Seq2SeqTrainingArguments(**{k: v for k, v in args_kw.items() if k in ta_params})

tr_kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
             data_collator=collator, compute_metrics=compute_metrics,
             callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
tr_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
tr_kw["processing_class" if "processing_class" in tr_params else "tokenizer"] = tok
trainer = Seq2SeqTrainer(**tr_kw)

# %% 13. Fine-tune
trainer.train(resume_from_checkpoint=RESUME_FROM)
final_dir = f"{OUT_DIR}/final_model"
trainer.save_model(final_dir)
tok.save_pretrained(final_dir)
pd.DataFrame(trainer.state.log_history).to_csv(f"{OUT_DIR}/training_log.csv", index=False)
print("Saved model to", final_dir)

# %% 14. Final evaluation of the fine-tuned model (best checkpoint is loaded)
print("\nTranslating test set with fine-tuned model ...")
test_out["finetuned"] = translate(test.src.tolist())
results.append(evaluate_system("fine-tuned", test_out.finetuned.tolist(), test.tgt.tolist(), test.domain.tolist()))
report = pd.concat(results, ignore_index=True)
print("\n=== TEST RESULTS ===")
print(report.round(2).to_string(index=False))
report.to_csv(f"{OUT_DIR}/test_metrics.csv", index=False)

# significance + error analysis
test_out["chrf_finetuned"] = sentence_chrf(test_out.finetuned, test.tgt)
if RUN_BASELINE:
    test_out["chrf_baseline"] = sentence_chrf(test_out.baseline, test.tgt)
    d, (lo, hi), p = paired_bootstrap(test_out.chrf_baseline.values, test_out.chrf_finetuned.values)
    print(f"\nPaired bootstrap (fine-tuned - baseline, sentence chrF++): mean diff={d:.2f}, "
          f"95% CI=[{lo:.2f}, {hi:.2f}], p~{p:.4f}")
    error_analysis(test_out, "baseline", f"{OUT_DIR}/worst_baseline.csv")
error_analysis(test_out, "finetuned", f"{OUT_DIR}/worst_finetuned.csv")
test_out.to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)

# %% 15. Optional COMET (semantic quality metric)
if RUN_COMET:
    from comet import download_model, load_from_checkpoint
    comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    for col in (["baseline"] if RUN_BASELINE else []) + ["finetuned"]:
        data_c = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(test_out.src, test_out[col], test_out.tgt)]
        print(col, "COMET:", comet.predict(data_c, batch_size=32, gpus=int(USE_CUDA)).system_score)

# %% 16. Inference helper
def translate_en_to_ur(sentences, beams=NUM_BEAMS):
    if isinstance(sentences, str):
        sentences = [sentences]
    return translate([clean_en(s) for s in sentences], num_beams=beams)


for s in ["How are you today?", "Knowledge is better than wealth.", "The meeting has been postponed until Monday."]:
    print(s, "->", translate_en_to_ur(s)[0])

# %% 17. Save everything you need to use the model on your own machine
import shutil

SAVE_ORIGINAL_PRETRAINED = True    # also keep the untouched NLLB base model (about 2.4 GB extra)
DELETE_CHECKPOINTS = True          # checkpoints include optimizer states and are huge; not needed locally

# (a) full-precision fine-tuned model (best checkpoint) -- already saved in step 13, re-save to be safe
full_dir = f"{OUT_DIR}/final_model"
model.save_pretrained(full_dir, safe_serialization=True)
tok.save_pretrained(full_dir)

# (b) half-precision copy: about half the size, same quality for inference, works on CPU and GPU
half_dir = f"{OUT_DIR}/final_model_fp16"
model.to(torch.float16).save_pretrained(half_dir, safe_serialization=True)
tok.save_pretrained(half_dir)
model.to(torch.float32)  # restore, in case you keep using the notebook

# (c) the original pretrained NLLB, if you want it locally for comparison
if SAVE_ORIGINAL_PRETRAINED:
    base_dir = f"{OUT_DIR}/nllb_pretrained_original"
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).save_pretrained(base_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(base_dir)

if DELETE_CHECKPOINTS:
    shutil.rmtree(f"{OUT_DIR}/checkpoints", ignore_errors=True)

# (d) zip for download (on Kaggle the files appear under the notebook's Output tab)
zip_base = os.path.join(os.path.dirname(OUT_DIR.rstrip("/")) or ".", "nllb_en_ur_finetuned_fp16")
zip_path = shutil.make_archive(zip_base, "zip", root_dir=half_dir)
print("Download this file:", zip_path, f"({os.path.getsize(zip_path) / 1e9:.2f} GB)")
print("Full-precision model folder:", full_dir)

# To reuse the model later:
#   tok = AutoTokenizer.from_pretrained(final_dir, src_lang="eng_Latn", tgt_lang="urd_Arab")
#   model = AutoModelForSeq2SeqLM.from_pretrained(final_dir)
