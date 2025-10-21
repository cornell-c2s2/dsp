# To train: python3 speaker_verify_mfcc.py train \ --pos data/pos --neg data/neg \ --min_precision 0.98
# To run: python3 speaker_verify_mfcc.py predict --folder testing



#!/usr/bin/env python3
import os, glob, json, argparse, warnings
import numpy as np
import soundfile as sf
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
from joblib import dump, load

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")

def list_audio_files(folder):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def load_audio(path, target_sr=8000):
    # soundfile handles most formats; fall back to librosa if needed
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:  # stereo -> mono
            y = np.mean(y, axis=1)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # trim long silences from both ends
    y, _ = librosa.effects.trim(y, top_db=30)
    return y.astype(np.float32), sr

def apply_vad(y, sr, top_db=30, min_speech_ms=120):
    # non-silent intervals; concatenate them (simple energy VAD)
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y  # fallback
    segs = []
    min_len = int(sr * (min_speech_ms/1000.0))
    for s, e in intervals:
        if (e - s) >= min_len:
            segs.append(y[s:e])
    if len(segs) == 0:
        return y
    return np.concatenate(segs)

def mfcc_cepstral_features(y, sr,
                           n_mfcc=20,
                           n_fft=512, win_length=400, hop_length=160):
    # Pre-emphasis to enhance high-freq (classic cepstral preprocessing)
    y = np.append(y[0], y[1:] - 0.97*y[:-1])

    # MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        htk=True
    )  # shape (n_mfcc, T)

    # Δ and Δ² (capture dynamics of the vocal tract)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    # Stack to (3*n_mfcc, T)
    X = np.vstack([mfcc, d1, d2])

    # Utterance-level pooling: robust stats
    def stats(mat):
        # mat: (F, T)
        feats = []
        feats.append(np.mean(mat, axis=1))
        feats.append(np.std(mat, axis=1))
        feats.append(np.median(mat, axis=1))
        feats.append(np.percentile(mat, 10, axis=1))
        feats.append(np.percentile(mat, 90, axis=1))
        return np.concatenate(feats, axis=0)  # (F*5,)
    pooled = stats(X)  # shape (3*n_mfcc*5,)
    return pooled.astype(np.float32)

def extract_feature_vector(path, sr=16000):
    y, sr = load_audio(path, target_sr=sr)
    y = apply_vad(y, sr, top_db=30, min_speech_ms=120)
    if len(y) < sr * 0.3:  # too little speech; pad or return zeros
        y = np.pad(y, (0, max(0, int(sr*0.3) - len(y))))
    return mfcc_cepstral_features(y, sr)

def build_dataset(pos_dir, neg_dir):
    pos_files = list_audio_files(pos_dir)
    neg_files = list_audio_files(neg_dir)
    X, y, paths = [], [], []
    for f in pos_files:
        X.append(extract_feature_vector(f))
        y.append(1)
        paths.append(f)
    for f in neg_files:
        X.append(extract_feature_vector(f))
        y.append(0)
        paths.append(f)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    return X, y, paths

def choose_threshold(y_true, scores, min_precision=0.97):
    """
    Pick a threshold that prioritizes high precision for the positive class.
    Falls back to F1-optimal threshold if required precision unattainable.
    """
    precision, recall, thresh = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # thresholds array is len-1 relative to precision/recall curves
    best = None
    for p, r, t in zip(precision[:-1], recall[:-1], thresh):
        if p >= min_precision:
            # pick the one with highest recall under the precision constraint
            if best is None or r > best[1]:
                best = (t, r, p)
    if best is not None:
        return float(best[0]), {"avg_precision": float(ap),
                                "picked_precision": float(best[2]),
                                "picked_recall": float(best[1]),
                                "strategy": f"precision>={min_precision}"}

    # fallback: maximize F1
    f1s = (2*precision*recall) / (precision+recall+1e-9)
    idx = int(np.nanargmax(f1s[:-1]))
    return float(thresh[idx]), {"avg_precision": float(ap),
                                "picked_precision": float(precision[idx]),
                                "picked_recall": float(recall[idx]),
                                "strategy": "max_f1_fallback"}

def train(pos_dir="data/pos", neg_dir="data/neg",
          model_out="models/mfcc_target_verifier.joblib",
          thresh_out="models/threshold.json",
          val_size=0.25, seed=42, min_precision=0.97):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    X, y, _ = build_dataset(pos_dir, neg_dir)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # Pipeline: standardize -> logistic regression -> probability calibration
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", base)
    ])
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    calibrated.fit(Xtr, ytr)

    # Validation scores
    va_scores = calibrated.predict_proba(Xva)[:, 1]
    thr, info = choose_threshold(yva, va_scores, min_precision=min_precision)

    # Final report at chosen threshold
    yhat = (va_scores >= thr).astype(int)
    print("\nValidation report @ chosen threshold:")
    print(classification_report(yva, yhat, digits=4))
    print("Threshold selection info:", info)

    # Persist
    dump(calibrated, model_out)
    with open(thresh_out, "w") as f:
        json.dump({"threshold": thr, "selection": info}, f, indent=2)
    print(f"\nSaved model -> {model_out}")
    print(f"Saved threshold -> {thresh_out}")

def load_model(model_path="models/mfcc_target_verifier.joblib",
               threshold_path="models/threshold.json"):
    model = load(model_path)
    with open(threshold_path, "r") as f:
        thr = json.load(f)["threshold"]
    return model, float(thr)

def predict_folder(folder="testing",
                   model_path="models/mfcc_target_verifier.joblib",
                   threshold_path="models/threshold.json",
                   sr=16000):
    model, thr = load_model(model_path, threshold_path)
    files = list_audio_files(folder)
    if not files:
        print(f"No audio files found in {folder}")
        return
    num_pos = 0
    num_neg = 0
    num_false_pos = 0
    num_false_neg = 0
    print(f"\nScoring {len(files)} file(s) in '{folder}' (threshold={thr:.4f})")
    for f in files:
        x = extract_feature_vector(f, sr=sr)[None, :]
        prob = float(model.predict_proba(x)[0, 1])
        label = "PASS (target)" if prob >= thr else "FAIL (not target)"
        print(f"{os.path.basename(f):<40}  score={prob:.4f}  ->  {label}")
        if "jackson" in os.path.basename(f):
            num_pos+=1
            if prob < thr:
                num_false_neg+=1
        else:
            num_neg+=1
            if prob >= thr:
                num_false_pos+=1
    total = num_pos + num_neg
    tp = num_pos - num_false_neg
    tn = num_neg - num_false_pos

    accuracy = (tp + tn) / total * 100
    false_pos_rate = num_false_pos / num_neg * 100
    false_neg_rate = num_false_neg / num_pos * 100
    precision = tp / (tp + num_false_pos) * 100
    recall = tp / (tp + num_false_neg) * 100

    print(f"Total samples     : {total}")
    print(f"Positive samples  : {num_pos} ({num_pos/total*100:.2f}%)")
    print(f"Negative samples  : {num_neg} ({num_neg/total*100:.2f}%)")
    print(f"False negatives   : {num_false_neg} ({false_neg_rate:.2f}%)")
    print(f"False positives   : {num_false_pos} ({false_pos_rate:.2f}%)")
    print(f"Accuracy          : {accuracy:.2f}%")
    print(f"Precision         : {precision:.2f}%")
    print(f"Recall (TPR)      : {recall:.2f}%")
        


def main():
    ap = argparse.ArgumentParser(description="MFCC-based target speaker verification")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train model on data/pos vs data/neg")
    ap_train.add_argument("--pos", default="data/pos")
    ap_train.add_argument("--neg", default="data/neg")
    ap_train.add_argument("--model", default="models/mfcc_target_verifier.joblib")
    ap_train.add_argument("--threshold", default="models/threshold.json")
    ap_train.add_argument("--val_size", type=float, default=0.25)
    ap_train.add_argument("--seed", type=int, default=42)
    ap_train.add_argument("--min_precision", type=float, default=0.97)

    ap_pred = sub.add_parser("predict", help="Score all files in testing/")
    ap_pred.add_argument("--folder", default="testing")
    ap_pred.add_argument("--model", default="models/mfcc_target_verifier.joblib")
    ap_pred.add_argument("--threshold", default="models/threshold.json")

    args = ap.parse_args()

    if args.cmd == "train":
        train(pos_dir=args.pos, neg_dir=args.neg,
              model_out=args.model, thresh_out=args.threshold,
              val_size=args.val_size, seed=args.seed,
              min_precision=args.min_precision)
    elif args.cmd == "predict":
        predict_folder(folder=args.folder,
                       model_path=args.model,
                       threshold_path=args.threshold)

if __name__ == "__main__":
    main()
