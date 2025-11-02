import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.mixture import GaussianMixture
from copy import deepcopy
import joblib

# =====================================
# CONFIGURATION
# =====================================
UBM_PATH = "models/ubm_gmm.joblib"
TARGET_CLIP = "data/garfield2.wav"
OUTPUT_PATH = "models/target_gmm.joblib"

SAMPLE_RATE = 16000
N_MFCC = 20
FRAME_LEN = 0.025
FRAME_STEP = 0.010
SEGMENT_DURATION = 3.0
OVERLAP = 1.0
RELEVANCE_FACTOR = 16


# =====================================
# FEATURE EXTRACTION
# =====================================
def extract_features(audio, sr):
    # Force mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) < int(FRAME_LEN * sr):
        return np.empty((0, N_MFCC))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=int(FRAME_LEN * sr),
        hop_length=int(FRAME_STEP * sr),
    ).T

    mfcc -= np.mean(mfcc, axis=0, keepdims=True)
    return mfcc


def segment_and_extract(file_path):
    y, sr = sf.read(file_path, always_2d=False)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Force mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    seg_len = int(SEGMENT_DURATION * sr)
    hop = int((SEGMENT_DURATION - OVERLAP) * sr)

    all_feats = []
    seg_count = 0

    for start in range(0, len(y) - seg_len, hop):
        end = start + seg_len
        seg = y[start:end]
        mfcc = extract_features(seg, sr)

        if mfcc.size > 0 and mfcc.ndim == 2:
            all_feats.append(mfcc)
            seg_count += 1
            if seg_count % 10 == 0:
                print(f"  processed {seg_count} segments (last shape={mfcc.shape})")
        else:
            print(
                f"⚠️ skipped segment at {start / sr:.2f}s (invalid shape {mfcc.shape})"
            )

    if not all_feats:
        raise ValueError("No valid MFCC features extracted!")

    all_feats = np.vstack(all_feats)
    print(f"✅ Extracted total MFCC feature matrix: {all_feats.shape}")
    return all_feats


# =====================================
# MAP ADAPTATION
# =====================================
def map_adapt_gmm(ubm, features, relevance_factor=16):
    adapted = deepcopy(ubm)
    n_components, n_features = ubm.means_.shape

    post = ubm.predict_proba(features)
    N = np.sum(post, axis=0) + 1e-8
    F = post.T @ features

    alpha = N / (N + relevance_factor)
    new_means = (alpha[:, None] * (F / N[:, None])) + (
        (1 - alpha)[:, None] * ubm.means_
    )
    adapted.means_ = new_means
    return adapted


# =====================================
# MAIN
# =====================================
print("Loading UBM...")
ubm = joblib.load(UBM_PATH)

print(f"Extracting MFCCs from {TARGET_CLIP}...")
target_feats = segment_and_extract(TARGET_CLIP)
print(f"Final feature shape (should be 2D): {target_feats.shape}")

print("Adapting UBM to target speaker...")
target_gmm = map_adapt_gmm(ubm, target_feats, RELEVANCE_FACTOR)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
joblib.dump(target_gmm, OUTPUT_PATH)
print(f"✅ Target GMM saved at {OUTPUT_PATH}")
