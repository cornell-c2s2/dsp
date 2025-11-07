import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from sklearn.mixture import GaussianMixture
from copy import deepcopy

from gmm_utils import (
    SAMPLE_RATE,
    extract_features_from_array,
    load_model,
    save_model,
)

# ill make these not hardcoded later i promise
UBM_PATH = "models/ubm_gmm.joblib"
TARGET_CLIP = "data/garfield2.wav"
OUTPUT_PATH = "models/target_gmm.joblib"

# how much of target clip to use by default (None = full clip)
DEFAULT_SECONDS_USED = 30
SEGMENT_DURATION = 3.0
OVERLAP = 1.0
RELEVANCE_FACTOR = 16

# Optional calibration constants (to be learned on dev set)
CAL_A, CAL_B, CAL_C = 1.0, 0.0, 0.0


def segment_and_extract(file_path, seconds_used=None):
    """Read audio, segment it, and extract MFCCs."""
    y, sr = sf.read(file_path, always_2d=False)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if seconds_used is not None:
        y = y[: int(seconds_used * sr)]
        print(f"🔹 Using first {seconds_used:.1f}s of audio")

    seg_len = int(SEGMENT_DURATION * sr)
    hop = int((SEGMENT_DURATION - OVERLAP) * sr)

    all_feats = []
    seg_count = 0

    for start in range(0, len(y) - seg_len, hop):
        end = start + seg_len
        seg = y[start:end]

        mfcc = extract_features_from_array(seg, sample_rate=sr)
        if mfcc.size > 0 and mfcc.ndim == 2:
            all_feats.append(mfcc)
            seg_count += 1
            if seg_count % 10 == 0:
                print(f"  processed {seg_count} segments (last shape={mfcc.shape})")
        else:
            print(f"skipped segment at {start / sr:.2f}s (invalid shape {mfcc.shape})")

    if not all_feats:
        raise ValueError("No valid MFCC features extracted")

    all_feats = np.vstack(all_feats)
    print(f"Extracted total MFCC feature matrix: {all_feats.shape}")
    return all_feats


def map_adapt_gmm(ubm, features, relevance_factor=16, fixed_alpha=0.7):
    """Mean-only MAP adaptation with fixed alpha to reduce duration bias."""
    adapted = deepcopy(ubm)
    n_components, n_features = ubm.means_.shape

    post = ubm.predict_proba(features)
    N = np.sum(post, axis=0) + 1e-8
    F = post.T @ features

    alpha = np.full(n_components, fixed_alpha)
    new_means = (alpha[:, None] * (F / N[:, None])) + (
        (1 - alpha)[:, None] * ubm.means_
    )
    adapted.means_ = new_means
    return adapted


def score_gmm(gmm_target, gmm_ubm, feats):
    """Compute per-frame mean log-likelihood ratio (LLR)."""
    ll_target = np.mean(gmm_target.score_samples(feats))
    ll_ubm = np.mean(gmm_ubm.score_samples(feats))
    n_frames = feats.shape[0]
    return (ll_target - ll_ubm), n_frames


def calibrated_score(raw_score, n_frames, A=1.0, B=0.0, C=0.0):
    """Duration-aware linear calibration."""
    return A * raw_score + B + C * np.log(n_frames + 1e-8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAP-adapt a UBM to a target speaker.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS_USED,
        help="Seconds of target audio to use (default: full clip)",
    )
    args = parser.parse_args()

    print("Loading UBM...")
    ubm = load_model(UBM_PATH)

    print(f"Extracting MFCCs from {TARGET_CLIP}...")
    target_feats = segment_and_extract(TARGET_CLIP, args.seconds)
    print(f"Final feature shape: {target_feats.shape}")

    print("Adapting UBM to target speaker...")
    target_gmm = map_adapt_gmm(ubm, target_feats, RELEVANCE_FACTOR)

    save_model(target_gmm, OUTPUT_PATH)
    print(f"Target GMM saved at {OUTPUT_PATH}")

    # Duration-robust scoring
    raw_score, n_frames = score_gmm(target_gmm, ubm, target_feats)
    cal_score = calibrated_score(raw_score, n_frames, CAL_A, CAL_B, CAL_C)

    print(f"\nRaw per-frame LLR:      {raw_score:.4f}")
    print(f"Calibrated duration LLR: {cal_score:.4f}")
