import os
import numpy as np
import librosa
import soundfile as sf
import joblib

import csv

# =====================================
# CONFIGURATION
# =====================================
TEST_DIR = "data/gmm_test/"  # folder with pos_*.wav and others
TARGET_MODEL = "models/target_gmm.joblib"
UBM_MODEL = "models/ubm_gmm.joblib"

SAMPLE_RATE = 16000
N_MFCC = 20
FRAME_LEN = 0.025
FRAME_STEP = 0.010
THRESHOLD = 0.94


# =====================================
# FEATURE EXTRACTION
# =====================================
def extract_features(file_path):
    audio, sr = sf.read(file_path)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=int(FRAME_LEN * SAMPLE_RATE),
        hop_length=int(FRAME_STEP * SAMPLE_RATE),
    ).T
    mfcc -= np.mean(mfcc, axis=0, keepdims=True)
    return mfcc


# =====================================
# LOAD MODELS
# =====================================
target_gmm = joblib.load(TARGET_MODEL)
ubm_gmm = joblib.load(UBM_MODEL)

# =====================================
# EVALUATION
# =====================================
results = []

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.lower().endswith(".wav"):
        continue

    fpath = os.path.join(TEST_DIR, fname)
    feats = extract_features(fpath)

    score_target = target_gmm.score(feats)
    score_ubm = ubm_gmm.score(feats)
    llr = score_target - score_ubm

    label = 1 if fname.startswith("pos_") else 0
    prediction = 1 if llr > THRESHOLD else 0
    correct = label == prediction

    results.append(
        {
            "file": fname,
            "llr": llr,
            "label": label,
            "pred": prediction,
            "correct": correct,
        }
    )

# =====================================
# REPORT
# =====================================
print(f"\n=== EVALUATION RESULTS ({len(results)} files) ===")
pos_scores = [r["llr"] for r in results if r["label"] == 1]
neg_scores = [r["llr"] for r in results if r["label"] == 0]

if pos_scores and neg_scores:
    print(f"Avg positive LLR: {np.mean(pos_scores):.3f}")
    print(f"Avg negative LLR: {np.mean(neg_scores):.3f}")

acc = np.mean([r["correct"] for r in results]) * 100
print(f"Accuracy @ threshold {THRESHOLD:.2f}: {acc:.2f}%\n")

for r in results:
    tag = "✅" if r["correct"] else "❌"
    print(
        f"{tag} {r['file']:<25}  LLR={r['llr']:+.3f}  Label={r['label']} Pred={r['pred']}"
    )

# Optionally save results to CSV

with open("results_llr.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n✅ Results saved to results_llr.csv")


# Find best threshold
llrs = np.array([r["llr"] for r in results])
labels = np.array([r["label"] for r in results])

thresholds = np.linspace(min(llrs), max(llrs), 200)
accs = [np.mean((llrs > t) == labels) for t in thresholds]
best_t = thresholds[np.argmax(accs)]
best_acc = max(accs) * 100
print(f"\n🔍 Best threshold = {best_t:.3f}  →  Accuracy = {best_acc:.2f}%")
