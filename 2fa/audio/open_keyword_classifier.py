# python3 open_keyword_classifier.py \ --pos data/pos \ --neg data/neg \ --test data/testing \ --model model.joblib
# Usage:
#   pip install librosa scikit-learn soundfile joblib
#   # also install ffmpeg on your system (brew/choco/apt, etc.)
#   python open_keyword_classifier.py --pos data/pos --neg data/neg --test data/testing --model model.joblib

import argparse
import glob
import os
import subprocess
import shutil
import joblib
import numpy as np
import librosa

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ============================== Audio loading (with OPUS support) ==============================

def _ffmpeg_load(path, sr=16000, mono=True):
    """
    Decode any audio (incl. .opus) to float32 PCM via ffmpeg at sample rate=sr.
    Returns: (y, sr)
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to read this file: " + path)

    channels = 1 if mono else 2
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), sr
    if not mono and channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, sr


def load_audio_any(path, sr=16000, mono=True):
    """
    Try librosa first (libsndfile/audioread). If it fails (e.g., OPUS not supported),
    fall back to ffmpeg decoding.
    """
    try:
        y, sr_out = librosa.load(path, sr=sr, mono=mono)
        return y, sr_out
    except Exception:
        return _ffmpeg_load(path, sr=sr, mono=mono)


# ============================== Feature extraction ==============================

def extract_features(path, sr=16000, n_mfcc=20, n_mels=64, hop_length=160, win_length=400):
    """
    Returns a fixed-length feature vector for an arbitrary-length clip.
    - MFCCs + deltas + deltadeltas: mean & std
    - Log-mel: mean & std
    - A few spectral/RMS stats
    """
    y, sr = load_audio_any(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros((n_mfcc * 6) + (n_mels * 2) + 10, dtype=np.float32)

    # (optional) per-clip normalization for compressed sources like OPUS
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    # pre-emphasis (helps speech formants)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=512, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, power=2.0
    )
    logmel = librosa.power_to_db(S + 1e-10)

    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(feat):
        return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])

    feats = [
        stats(mfcc),            # 2 * n_mfcc
        stats(d1),              # 2 * n_mfcc
        stats(d2),              # 2 * n_mfcc
        stats(logmel),          # 2 * n_mels
    ]

    # Spectral/RMS summary (means + stds)
    zcr  = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    sbw  = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    srf  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.90)
    rms  = librosa.feature.rms(y=y, hop_length=hop_length)
    addl = np.array([
        np.mean(zcr),  np.std(zcr),
        np.mean(sc),   np.std(sc),
        np.mean(sbw),  np.std(sbw),
        np.mean(srf),  np.std(srf),
        np.mean(rms),  np.std(rms),
    ]).astype(np.float32)

    feats.append(addl)
    return np.concatenate(feats).astype(np.float32)


# ============================== Data loading ==============================

def _all_files(root):
    return [p for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True) if os.path.isfile(p)]

def load_dataset(pos_dir, neg_dir):
    X, y = [], []
    for p in _all_files(pos_dir):
        X.append(extract_features(p)); y.append(1)
    for n in _all_files(neg_dir):
        X.append(extract_features(n)); y.append(0)
    if len(X) == 0:
        return np.empty((0, 1)), np.empty((0,), dtype=np.int64)
    return np.vstack(X), np.array(y, dtype=np.int64)

def load_testing(test_dir):
    files = sorted(_all_files(test_dir))
    if not files:
        return [], np.empty((0, 1))
    X = np.vstack([extract_features(f) for f in files])
    return files, X


# ============================== Train & Evaluate ==============================

def main():
    ap = argparse.ArgumentParser(description="Keyword spotter: detects the word 'open' (OPUS-ready)")
    ap.add_argument("--pos", default="data/pos", help="positive train dir (clips that contain 'open')")
    ap.add_argument("--neg", default="data/neg", help="negative train dir (clips without 'open')")
    ap.add_argument("--test", default="data/testing", help="testing dir (held-out clips)")
    ap.add_argument("--model", default="model.joblib", help="where to save the trained model")
    ap.add_argument("--C", type=float, default=2.0, help="logistic regression inverse regularization")
    args = ap.parse_args()

    print("Loading training data...")
    X, y = load_dataset(args.pos, args.neg)
    if X.shape[0] == 0:
        print("No training data found. Make sure --pos and --neg contain audio files.")
        return
    print(f"Train clips: {len(y)} | Pos={int(y.sum())} Neg={int((1 - y).sum())} | Dim={X.shape[1]}")

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ])

    print("Fitting model...")
    clf.fit(X, y)
    joblib.dump(clf, args.model)
    print(f"Saved model to {args.model}")

    preds_tr = clf.predict(X)
    print("\nTrain set (sanity) accuracy:", accuracy_score(y, preds_tr))
    print(classification_report(y, preds_tr, target_names=["negative", "positive"]))

    print("\nLoading testing data...")
    files, Xtest = load_testing(args.test)
    if len(files) == 0:
        print("No files found in testing directory.")
        return

    yhat = clf.predict(Xtest)
    yprob = clf.predict_proba(Xtest)[:, 1]

    print("\nPer-file predictions (1=contains 'open'):")
    for f, p, s in zip(files, yhat, yprob):
        print(f"{f}\t{p}\tprob={s:.3f}")


if __name__ == "__main__":
    main()
