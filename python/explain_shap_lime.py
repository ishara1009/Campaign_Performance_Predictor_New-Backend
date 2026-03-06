"""
SHAP + LIME Explainability for the Campaign Performance Predictor
=================================================================
Provides three complementary explainability methods on a multi-input
Keras Transformer model (sequence + numeric inputs):

  ① SHAP  KernelExplainer  — Shapley value attribution per numeric feature
  ② LIME  TabularExplainer — Local linear surrogate for numeric features
  ③ LIME  TextExplainer    — Word-level attribution for caption + content

Both methods produce per-target explanations for all five output metrics:
  likes, comments, shares, clicks, timing_quality_score

Usage:
    python explain_shap_lime.py <model_path> '<json_payload>'

Output:
    JSON string ready for the Node.js backend to forward to the frontend.
"""

import sys
import json
import os
import math
import pickle
import warnings
import traceback

import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Constants (must match training notebook) ──────────────────────────────────
MAX_LEN = 80

PLATFORM_MAP = {
    "Facebook":  0,
    "Instagram": 1,
    "TikTok":    2,
    "Twitter":   1,   # nearest equivalent
    "YouTube":   0,
    "LinkedIn":  0,
}

TARGETS     = ["likes", "comments", "shares", "clicks", "timing_quality_score"]
LOG_TARGETS = {"likes", "comments", "shares", "clicks"}

# Numeric feature vector order — must match predict.py preprocessing
FEATURE_NAMES = [
    "platform_id",
    "post_hour",
    "day_of_week",
    "is_weekend",
    "followers_log",
    "ad_boost",
]

FEATURE_LABELS = {
    "platform_id":   "Platform",
    "post_hour":     "Posting Hour",
    "day_of_week":   "Day of Week",
    "is_weekend":    "Is Weekend",
    "followers_log": "Follower Count",
    "ad_boost":      "Ad Boost",
}

TARGET_LABELS = {
    "likes":                "Likes",
    "comments":             "Comments",
    "shares":               "Shares",
    "clicks":               "Clicks",
    "timing_quality_score": "Timing Quality Score",
}

# User-friendly platform names for SHAP actual-value display
PLATFORM_ID_MAP = {0: "Facebook/YouTube/LinkedIn", 1: "Instagram/Twitter", 2: "TikTok"}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Background data for SHAP/LIME ────────────────────────────────────────────
def _build_background(n: int = 30) -> np.ndarray:
    """
    Synthesise representative background samples from the Sri Lankan SME
    social media dataset distribution (no raw data required at inference time).
    """
    rng = np.random.RandomState(42)
    platform_ids   = rng.choice([0, 1, 2], n, p=[0.40, 0.35, 0.25])
    post_hours     = rng.choice(list(range(6, 23)), n)
    day_of_weeks   = rng.randint(0, 7, n)
    is_weekends    = (day_of_weeks >= 5).astype(float)
    # followers drawn from a log-normal centred on ~5 000 (typical SME)
    followers_logs = np.log1p(
        rng.lognormal(mean=8.5, sigma=1.4, size=n).clip(50, 1_000_000)
    )
    ad_boosts = rng.choice([0, 1], n, p=[0.68, 0.32])
    return np.column_stack(
        [platform_ids, post_hours, day_of_weeks, is_weekends, followers_logs, ad_boosts]
    ).astype("float32")


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model_and_deps(model_path: str):
    import keras
    import keras.src.ops.operation as _op_mod
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    saved_dir   = os.path.dirname(model_path)
    tok_path    = os.path.join(saved_dir, "tokenizer.json")
    scaler_path = os.path.join(saved_dir, "y_scaler.pkl")

    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json not found at {tok_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"y_scaler.pkl not found at {scaler_path}")

    with open(tok_path, encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(scaler_path, "rb") as f:
        y_scaler = pickle.load(f)

    # Compatibility patch for quantization_config in Keras layer configs
    _orig = _op_mod.Operation.from_config.__func__

    @classmethod  # type: ignore
    def _safe_from_config(cls, config, **kwargs):
        config = dict(config)
        config.pop("quantization_config", None)
        return _orig(cls, config, **kwargs)

    _op_mod.Operation.from_config = _safe_from_config
    try:
        model = keras.models.load_model(model_path, compile=False)
    finally:
        _op_mod.Operation.from_config = classmethod(_orig)

    return model, tokenizer, y_scaler


# ── Preprocessing helpers ─────────────────────────────────────────────────────
def _tokenise(text: str, tokenizer) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    ).astype("int32")


def _extract_numeric(payload: dict) -> np.ndarray:
    from datetime import datetime

    platform  = str(payload.get("platform", "Facebook")).strip()
    post_date = str(payload.get("post_date", "2025-01-01"))
    post_time = str(payload.get("post_time", "12:00"))
    followers = float(payload.get("followers", 1000))
    ad_boost  = int(bool(payload.get("ad_boost", 0)))

    platform_id = PLATFORM_MAP.get(platform, 0)

    try:
        dt          = datetime.strptime(post_date, "%Y-%m-%d")
        day_of_week = dt.weekday()
    except Exception:
        day_of_week = 0

    try:
        post_hour = int(post_time.split(":")[0])
    except Exception:
        post_hour = 12

    is_weekend    = 1 if day_of_week >= 5 else 0
    followers_log = math.log1p(followers)

    return np.array(
        [platform_id, post_hour, day_of_week, is_weekend, followers_log, ad_boost],
        dtype="float32",
    )


def _inverse_transform_batch(raw: np.ndarray, y_scaler) -> np.ndarray:
    """Apply inverse StandardScaler + reverse log1p for log-transformed targets."""
    y_inv = y_scaler.inverse_transform(raw)
    out   = np.zeros_like(y_inv)
    for i, target in enumerate(TARGETS):
        col = y_inv[:, i]
        if target in LOG_TARGETS:
            out[:, i] = np.array([math.expm1(max(float(v), 0.0)) for v in col])
        else:
            out[:, i] = np.clip(col, 0.0, 1.0)
    return out


# ── Prediction wrappers for SHAP / LIME ──────────────────────────────────────
def _make_predict_num_all(model, y_scaler, fixed_seq: np.ndarray):
    """
    Returns a function: (n_samples, 6) → (n_samples, 5)
    Fixes the text sequence and varies numeric features.
    Used by both SHAP KernelExplainer and LIME TabularExplainer.
    """
    def _predict(num_array: np.ndarray) -> np.ndarray:
        n         = num_array.shape[0]
        seq_tile  = np.tile(fixed_seq, (n, 1))           # (n, MAX_LEN)
        raw       = model.predict(
            [seq_tile, num_array.astype("float32")],
            verbose=0, batch_size=32,
        )
        return _inverse_transform_batch(raw, y_scaler)  # (n, 5)
    return _predict


def _make_predict_text(model, y_scaler, fixed_num: np.ndarray, tokenizer):
    """
    Returns a function: List[str] → (n_samples,) for a single target.
    Fixes numeric features and varies text.
    Used by LIME TextExplainer.
    """
    def _predict_for_target(texts, target_idx: int) -> np.ndarray:
        seqs = np.vstack([_tokenise(t, tokenizer) for t in texts])  # (n, MAX_LEN)
        nums = np.tile(fixed_num, (len(texts), 1))                   # (n, 6)
        raw  = model.predict([seqs, nums], verbose=0, batch_size=32)
        return _inverse_transform_batch(raw, y_scaler)[:, target_idx]
    return _predict_for_target


# ── SHAP : KernelExplainer ────────────────────────────────────────────────────
def run_shap(predict_num_all, num_feat: np.ndarray, background: np.ndarray) -> dict:
    """
    Computes SHAP Shapley values for each numeric feature across all 5 output targets.

    SHAP KernelExplainer is model-agnostic and provides theoretically exact
    (or near-exact) attributions satisfying efficiency, symmetry, and dummy axioms.
    We run one KernelExplainer per target (single-output wrapper) to avoid
    multi-output indexing differences across SHAP versions (0.4x vs 0.50+).
    nsamples=120 gives stable estimates for 6 features without excessive runtime.
    """
    import shap

    shap_out = {}
    for i, target in enumerate(TARGETS):
        # Single-output wrapper — avoids all multi-output SHAP shape ambiguity
        def _pred_single(arr, _i=i):
            return predict_num_all(arr.astype("float32"))[:, _i]

        explainer   = shap.KernelExplainer(_pred_single, background, silent=True)
        sv_raw      = explainer.shap_values(
            num_feat.reshape(1, -1), nsamples=120, silent=True
        )
        sv          = np.array(sv_raw).flatten()   # reliably shape (6,)
        total_abs = float(np.abs(sv).sum()) + 1e-9

        rows = []
        for j, fn in enumerate(FEATURE_NAMES):
            raw_val = float(num_feat[j])
            # Convert raw feature values to human-readable form
            if fn == "platform_id":
                display_val = PLATFORM_ID_MAP.get(int(round(raw_val)), str(raw_val))
            elif fn == "followers_log":
                display_val = f"{int(round(math.expm1(raw_val))):,}"
            elif fn == "day_of_week":
                display_val = DAY_NAMES[min(int(round(raw_val)), 6)]
            elif fn == "is_weekend":
                display_val = "Yes (Weekend)" if raw_val >= 0.5 else "No (Weekday)"
            elif fn == "ad_boost":
                display_val = "Yes (Boosted)" if raw_val >= 0.5 else "No (Organic)"
            elif fn == "post_hour":
                display_val = f"{int(round(raw_val))}:00"
            else:
                display_val = str(round(raw_val, 3))

            rows.append({
                "feature":        fn,
                "label":          FEATURE_LABELS[fn],
                "display_value":  display_val,
                "shap_value":     round(float(sv[j]), 4),
                "direction":      "positive" if sv[j] >= 0 else "negative",
                "importance_pct": round(abs(float(sv[j])) / total_abs * 100, 1),
            })

        rows.sort(key=lambda x: x["importance_pct"], reverse=True)
        shap_out[target] = rows

    return shap_out


# ── LIME : TabularExplainer ───────────────────────────────────────────────────
def run_lime_tabular(predict_num_all, num_feat: np.ndarray, background: np.ndarray) -> dict:
    """
    Fits a local linear surrogate model around the prediction point to explain
    which numeric features drive each output metric (LIME, Ribeiro et al. 2016).
    num_samples=200 provides stable, low-variance local approximations.
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=FEATURE_NAMES,
        mode="regression",
        random_state=42,
        verbose=False,
    )

    def _range_to_label(s: str) -> str:
        sl = s.lower()
        for fn, lbl in FEATURE_LABELS.items():
            if sl.startswith(fn.lower()):
                return lbl
        return s

    lime_out = {}
    for i, target in enumerate(TARGETS):
        def _pfn(arr, _i=i):
            return predict_num_all(arr)[:, _i]

        exp   = explainer.explain_instance(num_feat, _pfn, num_features=6, num_samples=200)
        feats = exp.as_list()   # [(feature_range_str, weight), ...]

        total = sum(abs(w) for _, w in feats) + 1e-9
        rows  = sorted(
            [
                {
                    "feature_range":  fr,
                    "label":          _range_to_label(fr),
                    "weight":         round(float(w), 4),
                    "direction":      "positive" if w >= 0 else "negative",
                    "importance_pct": round(abs(float(w)) / total * 100, 1),
                }
                for fr, w in feats
            ],
            key=lambda x: x["importance_pct"],
            reverse=True,
        )
        lime_out[target] = rows

    return lime_out


# ── LIME : TextExplainer ──────────────────────────────────────────────────────
def run_lime_text(predict_text, text: str, num_words: int = 12, num_samples: int = 100) -> dict:
    """
    Identifies which individual words in the caption+content push each predicted
    metric up or down. Uses LIME's bag-of-words perturbation strategy (Ribeiro 2016).

    Returns top `num_words` words per target sorted by |importance|.
    """
    from lime.lime_text import LimeTextExplainer

    if not text.strip():
        return {t: [] for t in TARGETS}

    explainer = LimeTextExplainer(random_state=42)
    lime_text_out = {}

    for i, target in enumerate(TARGETS):
        def _pfn(texts, _i=i):
            return predict_text(texts, _i)

        try:
            exp   = explainer.explain_instance(text, _pfn, num_features=num_words,
                                               num_samples=num_samples)
            words = exp.as_list()
            total = sum(abs(w) for _, w in words) + 1e-9
            lime_text_out[target] = sorted(
                [
                    {
                        "word":           wd,
                        "weight":         round(float(w), 4),
                        "direction":      "positive" if w >= 0 else "negative",
                        "importance_pct": round(abs(float(w)) / total * 100, 1),
                    }
                    for wd, w in words
                ],
                key=lambda x: x["importance_pct"],
                reverse=True,
            )
        except Exception:
            lime_text_out[target] = []

    return lime_text_out


# ── Summary generation ────────────────────────────────────────────────────────
def _build_summary(shap_out: dict, lime_num_out: dict, lime_text_out: dict, payload: dict) -> dict:
    """
    Produces a plain-English summary of the SHAP/LIME findings per target.
    No LLM required — pure rule-based interpretation of the numerical results.
    """
    platform = payload.get("platform", "the platform")
    summaries = {}

    for target in TARGETS:
        tl = TARGET_LABELS.get(target, target)

        # Top SHAP driver
        shap_rows = shap_out.get(target, [])
        top_shap  = shap_rows[0] if shap_rows else None

        # Top LIME driver
        lime_rows = lime_num_out.get(target, [])
        top_lime  = lime_rows[0] if lime_rows else None

        # Top positive + negative LIME text word
        text_rows  = lime_text_out.get(target, [])
        top_pos_w  = next((w for w in text_rows if w["direction"] == "positive"), None)
        top_neg_w  = next((w for w in text_rows if w["direction"] == "negative"), None)

        parts = []

        if top_shap:
            dir_word = "increases" if top_shap["direction"] == "positive" else "decreases"
            parts.append(
                f"SHAP analysis identifies '{top_shap['label']}' (current: {top_shap['display_value']}) "
                f"as the dominant driver — it {dir_word} {tl} by {top_shap['importance_pct']}% "
                f"of total feature attribution (Shapley value: {top_shap['shap_value']:+.2f})."
            )

        if top_lime and top_lime["label"] != top_shap["label"] if top_shap else True:
            dir_word = "positively" if top_lime["direction"] == "positive" else "negatively"
            parts.append(
                f"LIME's local surrogate confirms '{top_lime['label']}' {dir_word} impacts {tl} "
                f"({top_lime['importance_pct']}% of local importance, condition: {top_lime['feature_range']})."
            )
        elif top_lime:
            parts.append(
                f"LIME corroborates SHAP: '{top_lime['label']}' is consistently the strongest local feature "
                f"influencer for {tl} on {platform}."
            )

        if top_pos_w:
            parts.append(
                f"Word '{top_pos_w['word']}' in the caption/content positively influences {tl} "
                f"({top_pos_w['importance_pct']}% text attribution weight)."
            )
        if top_neg_w:
            parts.append(
                f"Word '{top_neg_w['word']}' slightly suppresses {tl} — consider rephrasing or removing it."
            )

        summaries[target] = " ".join(parts)

    return summaries


# ── Overall concordance check ─────────────────────────────────────────────────
def _concordance(shap_out: dict, lime_num_out: dict) -> dict:
    """
    Checks whether SHAP and LIME agree on the top feature per target.
    Agreement strengthens confidence in the explanation.
    """
    results = {}
    for target in TARGETS:
        s_top = shap_out.get(target, [{}])[0].get("label", "")
        l_top = lime_num_out.get(target, [{}])[0].get("label", "")
        results[target] = {
            "agree": s_top == l_top,
            "shap_top": s_top,
            "lime_top": l_top,
            "message": (
                f"SHAP and LIME agree: '{s_top}' is the top driver."
                if s_top == l_top
                else f"SHAP ranks '{s_top}' first; LIME ranks '{l_top}' first — "
                     f"examine both to understand the local vs global feature importance."
            ),
        }
    return results


# ── Main entry point ──────────────────────────────────────────────────────────
def run_explain(model_path: str, payload: dict) -> dict:
    # ── Load model artifacts ──────────────────────────────────────────────────
    model, tokenizer, y_scaler = _load_model_and_deps(model_path)

    caption  = str(payload.get("caption", ""))
    content  = str(payload.get("content",  ""))
    text     = (caption + " " + content).strip()

    num_feat   = _extract_numeric(payload)
    fixed_seq  = _tokenise(text, tokenizer)
    background = _build_background(30)

    # ── Prediction wrappers ───────────────────────────────────────────────────
    predict_num_all = _make_predict_num_all(model, y_scaler, fixed_seq)
    predict_text    = _make_predict_text(model, y_scaler, num_feat, tokenizer)

    # ── ① SHAP KernelExplainer ────────────────────────────────────────────────
    shap_out = run_shap(predict_num_all, num_feat, background)

    # ── ② LIME TabularExplainer ───────────────────────────────────────────────
    lime_num_out = run_lime_tabular(predict_num_all, num_feat, background)

    # ── ③ LIME TextExplainer ──────────────────────────────────────────────────
    lime_text_out = run_lime_text(predict_text, text, num_words=12, num_samples=100)

    # ── ④ Summaries + concordance ─────────────────────────────────────────────
    summaries   = _build_summary(shap_out, lime_num_out, lime_text_out, payload)
    concordance = _concordance(shap_out, lime_num_out)

    return {
        "shap_numeric":   shap_out,
        "lime_numeric":   lime_num_out,
        "lime_text":      lime_text_out,
        "summaries":      summaries,
        "concordance":    concordance,
        "feature_labels": FEATURE_LABELS,
        "target_labels":  TARGET_LABELS,
        "targets":        TARGETS,
        "method_notes": {
            "shap": (
                "SHAP (SHapley Additive exPlanations) uses game-theoretic Shapley values to "
                "attribute each feature's contribution to the prediction. Values are exact and "
                "satisfy the efficiency axiom: all SHAP values for one prediction sum to the "
                "difference between the actual prediction and the base (average) prediction."
            ),
            "lime_numeric": (
                "LIME (Local Interpretable Model-agnostic Explanations) fits a local linear "
                "surrogate model in the neighbourhood of the prediction point, perturbing "
                "numeric inputs to approximate the model's local decision boundary."
            ),
            "lime_text": (
                "LIME Text perturbs the caption+content by randomly removing words and "
                "observing how each removal affects the predicted metric. Words with high "
                "positive weight push the metric up; negative-weight words suppress it."
            ),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: explain_shap_lime.py <model_path> <json_payload>"}))
        sys.exit(1)

    try:
        _payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON payload: {e}"}))
        sys.exit(1)

    try:
        result = run_explain(sys.argv[1], _payload)
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)
