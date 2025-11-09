# ...existing code...
import os
import numpy as np
import librosa
import soundfile as sf
import joblib

SAMPLE_RATE = 16000
N_MFCC = 13
FRAME_LEN = 0.025
FRAME_STEP = 0.010


def sliding_cmvn(feats, win_size=300, eps=1e-8):
    """Apply sliding-window cepstral mean/variance normalization."""
    n_frames, n_feats = feats.shape
    out = np.empty_like(feats)
    for t in range(n_frames):
        start = max(0, t - win_size // 2)
        end = min(n_frames, t + win_size // 2)
        window = feats[start:end, :]
        mu = window.mean(axis=0, keepdims=True)
        sigma = window.std(axis=0, keepdims=True)
        out[t : t + 1, :] = (feats[t : t + 1, :] - mu) / (sigma + eps)
    return out


def extract_features_from_array(
    audio,
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    frame_len=FRAME_LEN,
    frame_step=FRAME_STEP,
):
    """Extract MFCC + Δ + ΔΔ features (mean-normalized) from a numpy array.

    Parameters:
    - audio: 1-D or 2-D numpy array containing audio samples. If 2-D, channels
      are averaged to mono.
    - sample_rate: sample rate of the provided audio array.

    Returns:
    - feats: (n_frames, n_mfcc*3) feature matrix (mfcc, delta, delta-delta)
    """
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) < int(frame_len * sample_rate):
        return np.empty((0, n_mfcc * 3))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=int(frame_len * sample_rate),
        hop_length=int(frame_step * sample_rate),
    )

    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feats = np.vstack([mfcc, delta1, delta2]).T
    feats = sliding_cmvn(feats, win_size=300)
    return feats


def extract_features(
    file_path,
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    frame_len=FRAME_LEN,
    frame_step=FRAME_STEP,
):
    """Read an audio file and extract MFCC features by delegating to
    `extract_features_from_array`.

    Keeping this wrapper preserves the original, file-path based API for
    callers that pass filenames.
    """
    audio, sr = sf.read(file_path)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    return extract_features_from_array(
        audio, sample_rate=sr, n_mfcc=n_mfcc, frame_len=frame_len, frame_step=frame_step
    )


def load_model(path):
    """Load a joblib model."""
    return joblib.load(path)


def save_model(model, path):
    """Save a joblib model."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)


def score_models(target_model, ubm_model, feats):
    """Return (score_target, score_ubm)."""
    return target_model.score(feats), ubm_model.score(feats)


def evaluate_dir(test_dir, target_model, ubm_model, threshold):
    """Evaluate all wav files in test_dir and return results list."""
    results = []
    for fname in sorted(os.listdir(test_dir)):
        print(f"Scoring file: {fname}")
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(test_dir, fname)
        feats = extract_features(fpath)
        score_t, score_u = score_models(target_model, ubm_model, feats)
        llr = score_t - score_u
        label = 1 if fname.startswith("pos") else 0
        pred = 1 if llr > threshold else 0
        results.append(
            {
                "file": fname,
                "llr": llr,
                "label": label,
                "pred": pred,
                "correct": label == pred,
            }
        )
    return results


def predict_file(file_path, target_model, ubm_model, threshold=None):
    """Score a single wav file.

    Returns a dict with keys:
      - file: filename
      - llr: log-likelihood ratio (target - ubm)
      - score_target: raw target model score
      - score_ubm: raw ubm model score
      - pred: 0/1 prediction when `threshold` is provided, else None
      - label: None (unknown for single file)

    `threshold` is optional; if provided, `pred` will be set.
    """
    fname = os.path.basename(file_path)
    feats = extract_features(file_path)
    score_t, score_u = score_models(target_model, ubm_model, feats)
    llr = score_t - score_u
    pred = None
    if threshold is not None:
        pred = 1 if llr > threshold else 0

    out = {
        "file": fname,
        "llr": llr,
        "score_target": score_t,
        "score_ubm": score_u,
        "pred": pred,
        "label": None,
    }

    return out
