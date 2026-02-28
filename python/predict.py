"""
Transformer model inference script.
Preprocessing exactly matches the training notebook (reach.ipynb).

Usage: python predict.py <model_path> '<json_payload>'
Outputs a JSON string with predictions.
"""

import sys
import json
import os
import math
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Constants (must match training notebook) ──────────────────────────────────
MAX_LEN   = 80
MAX_VOCAB = 30000

# Platform map from training data (pandas category codes, sorted alphabetically)
# {0: 'Facebook', 1: 'Instagram', 2: 'TikTok'}
PLATFORM_MAP = {
    'Facebook':   0,
    'Instagram':  1,
    'TikTok':     2,
}

TARGETS = ['likes', 'comments', 'shares', 'clicks', 'timing_quality_score']
# First 4 targets were log1p-transformed before StandardScaler during training
LOG_TARGETS = {'likes', 'comments', 'shares', 'clicks'}


def preprocess(payload: dict, tokenizer, num_scaler=None):
    """
    Builds the two model inputs:
      1.  seq  – (1, MAX_LEN)  tokenised + padded text
      2.  num  – (1, 6)        numeric feature vector

    num_features order (from notebook):
      platform_id, post_hour, day_of_week, is_weekend, followers_log, ad_boost
    """
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    caption   = str(payload.get('caption', ''))
    content   = str(payload.get('content', ''))
    platform  = str(payload.get('platform', 'Facebook')).strip()
    post_date = str(payload.get('post_date', '2025-01-01'))
    post_time = str(payload.get('post_time', '12:00'))
    followers = float(payload.get('followers', 1000))
    ad_boost  = int(bool(payload.get('ad_boost', 0)))

    # ── Text ─────────────────────────────────────────────────────────────────
    text = (caption + ' ' + content).strip()
    seq  = tokenizer.texts_to_sequences([text])
    seq  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    # shape: (1, MAX_LEN)

    # ── Platform ID ──────────────────────────────────────────────────────────
    platform_id = PLATFORM_MAP.get(platform, 0)   # default to Facebook (0)

    # ── Time features ────────────────────────────────────────────────────────
    try:
        from datetime import datetime
        dt          = datetime.strptime(post_date, '%Y-%m-%d')
        day_of_week = dt.weekday()    # 0=Mon … 6=Sun
    except Exception:
        day_of_week = 0

    try:
        post_hour = int(post_time.split(':')[0])
    except Exception:
        post_hour = 12

    is_weekend = 1 if day_of_week >= 5 else 0

    # ── Followers ────────────────────────────────────────────────────────────
    followers_log = math.log1p(followers)

    # ── Numeric feature vector ────────────────────────────────────────────────
    num = np.array(
        [[platform_id, post_hour, day_of_week, is_weekend, followers_log, ad_boost]],
        dtype='float32'
    )  # shape: (1, 6)

    return seq.astype('int32'), num


def predict(model_path: str, payload: dict) -> dict:
    import pickle
    import numpy as np
    import keras

    saved_dir   = os.path.dirname(model_path)
    tok_path    = os.path.join(saved_dir, 'tokenizer.json')
    scaler_path = os.path.join(saved_dir, 'y_scaler.pkl')

    # ── Load tokenizer ────────────────────────────────────────────────────────
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f'tokenizer.json not found at {tok_path}')

    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(tok_path, encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())

    # ── Load y_scaler ─────────────────────────────────────────────────────────
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'y_scaler.pkl not found at {scaler_path}')

    with open(scaler_path, 'rb') as f:
        y_scaler = pickle.load(f)

    # ── Load Transformer model (version-compatibility patch) ───────────────────
    # Model was saved with a Keras dev build that adds `quantization_config`
    # to every layer's get_config(). Patch the Operation base class's
    # from_config so it strips that key before calling cls(**config).
    import keras.src.ops.operation as _op_mod

    _orig_from_config = _op_mod.Operation.from_config.__func__

    @classmethod  # type: ignore
    def _safe_from_config(cls, config, **kwargs):
        config = dict(config)
        config.pop('quantization_config', None)
        return _orig_from_config(cls, config, **kwargs)

    _op_mod.Operation.from_config = _safe_from_config

    try:
        model = keras.models.load_model(model_path, compile=False)
    finally:
        _op_mod.Operation.from_config = classmethod(_orig_from_config)

    # ── Preprocess inputs ─────────────────────────────────────────────────────
    seq, num = preprocess(payload, tokenizer)

    # ── Run inference ─────────────────────────────────────────────────────────
    raw = model.predict([seq, num], verbose=0)   # shape: (1, 5)

    # ── Inverse-transform outputs ─────────────────────────────────────────────
    y_inv = y_scaler.inverse_transform(raw)      # back to log1p scale

    result = {}
    for i, target in enumerate(TARGETS):
        val = float(y_inv[0, i])
        if target in LOG_TARGETS:
            # Reverse the log1p applied during training
            val = math.expm1(max(val, 0.0))
        else:
            # timing_quality_score – clamp to [0, 1]
            val = min(max(val, 0.0), 1.0)
        result[target] = round(val, 4 if target == 'timing_quality_score' else 2)

    return result


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: predict.py <model_path> <json_payload>'}))
        sys.exit(1)

    model_path_arg = sys.argv[1]
    payload_arg    = sys.argv[2]

    try:
        payload_dict = json.loads(payload_arg)
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON payload: {str(e)}'}))
        sys.exit(1)

    try:
        results = predict(model_path_arg, payload_dict)
        print(json.dumps(results))
    except Exception as exc:
        print(json.dumps({'error': str(exc)}))
        sys.exit(1)
