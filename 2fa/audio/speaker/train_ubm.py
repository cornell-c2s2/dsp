import os
import random
import numpy as np
import pandas as pd
from gmm_utils import (
    extract_features,
    save_model,
    N_MFCC,
    SAMPLE_RATE,
    FRAME_LEN,
    FRAME_STEP,
)
from sklearn.mixture import GaussianMixture

# data isn't in the repo due to size, trained w/
# can get it here: https://commonvoice.mozilla.org/en/datasets
# paths hardcoded because I'm lazy
DATA_ROOT = "data/ubm/cv-corpus-22.0-delta-2025-06-20/en"
CLIPS_DIR = os.path.join(DATA_ROOT, "clips")
TSV_FILE = os.path.join(DATA_ROOT, "clip_durations.tsv")  # update name if different
N_MIXTURES = 16
MAX_FILES = 3_000  # number of files to sample (3000 seems fine for now)

df = pd.read_csv(TSV_FILE, sep="\t")
file_list = (
    df["filename"].tolist() if "filename" in df.columns else df.iloc[:, 0].tolist()
)

# Randomly sample subset for training
random.shuffle(file_list)
file_list = file_list[:MAX_FILES]

print(f"Using {len(file_list)} audio clips for training...")


features = []
for i, fname in enumerate(file_list):
    fpath = os.path.join(CLIPS_DIR, fname)
    try:
        feats = extract_features(
            fpath,
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            frame_len=FRAME_LEN,
            frame_step=FRAME_STEP,
        )
    except Exception as e:
        print(f"Skipping {fpath}: {e}")
        feats = None

    if feats is not None:
        features.append(feats)
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(file_list)} files...")

all_feats = np.vstack(features)
print(f"Total feature vectors: {all_feats.shape}")

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
save_model(ubm, "models/ubm_gmm.joblib")
print("UBM training complete, saved to models/ubm_gmm.joblib")
