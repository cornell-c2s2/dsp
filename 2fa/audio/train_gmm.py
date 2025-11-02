import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.mixture import GaussianMixture
import joblib

# ====================================
# CONFIGURATION
# ====================================
DATA_ROOT = "data/ubm/cv-corpus-22.0-delta-2025-06-20/en"
CLIPS_DIR = os.path.join(DATA_ROOT, "clips")
TSV_FILE = os.path.join(DATA_ROOT, "clip_durations.tsv")  # update name if different
N_MFCC = 20
N_MIXTURES = 64
SAMPLE_RATE = 16000
FRAME_LEN = 0.025
FRAME_STEP = 0.010
MAX_FILES = 3000  # number of files to sample (adjust based on RAM)

# ====================================
# LOAD FILE LIST
# ====================================
df = pd.read_csv(TSV_FILE, sep="\t")
file_list = (
    df["filename"].tolist() if "filename" in df.columns else df.iloc[:, 0].tolist()
)

# Randomly sample subset for training
random.shuffle(file_list)
file_list = file_list[:MAX_FILES]

print(f"Using {len(file_list)} audio clips for training...")


# ====================================
# FEATURE EXTRACTION
# ====================================
def extract_features(file_path):
    try:
        audio, sr = sf.read(file_path)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=int(FRAME_LEN * SAMPLE_RATE),
            hop_length=int(FRAME_STEP * SAMPLE_RATE),
        ).T

        mfcc -= np.mean(mfcc, axis=0, keepdims=True)
        return mfcc
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None


features = []
for i, fname in enumerate(file_list):
    fpath = os.path.join(CLIPS_DIR, fname)
    feats = extract_features(fpath)
    if feats is not None:
        features.append(feats)
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(file_list)} files...")

all_feats = np.vstack(features)
print(f"Total feature vectors: {all_feats.shape}")

# ====================================
# TRAIN THE UBM
# ====================================
print("Training UBM GMM...")
ubm = GaussianMixture(
    n_components=N_MIXTURES,
    covariance_type="diag",
    max_iter=300,
    n_init=2,
    verbose=1,
    random_state=42,
).fit(all_feats)

os.makedirs("models", exist_ok=True)
joblib.dump(ubm, "models/ubm_gmm.joblib")
print("✅ UBM training complete! Saved as models/ubm_gmm.joblib")
