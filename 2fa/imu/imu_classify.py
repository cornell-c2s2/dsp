#!/usr/bin/env python3
# imu_classify.py
# Usage:
#   Train:   python imu_classify.py train --train_dir data/train --model models/imu_rf.joblib
#   Predict: python imu_classify.py predict --folder data/test --model models/imu_rf.joblib
# Expected layout:
# data/train/
#   clap/*.csv
#   wave/*.csv
#   idle/*.csv
#
# Each CSV: columns like time,ax,ay,az,gx,gy,gz (time in seconds or ms)

import os, glob, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load

warnings.filterwarnings("ignore", category=UserWarning)
IMU_EXTS = (".csv", ".tsv")

def list_imu_files(folder):
    files = []
    for ext in IMU_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def read_imu_csv(path):
    # Try common headers; allow both comma and tab
    sep = "," if path.endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep)
    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # Required axes: ax, ay, az, gx, gy, gz
    # Accept variants like acc_x, accelx, gyro_x, wx, etc.
    def find(names, candidates):
        for n in names:
            if n in cols: return cols[n]
        for cand in candidates:
            if cand in df.columns: return cand
        return None

    ax = find(["ax", "accx", "acc_x", "accelx"], ["ax","acc_x"])
    ay = find(["ay", "accy", "acc_y", "accely"], ["ay","acc_y"])
    az = find(["az", "accz", "acc_z", "accelz"], ["az","acc_z"])
    gx = find(["gx", "gyrox", "gyr_x", "wx"], ["gx","gyr_x"])
    gy = find(["gy", "gyroy", "gyr_y", "wy"], ["gy","gyr_y"])
    gz = find(["gz", "gyroz", "gyr_z", "wz"], ["gz","gyr_z"])

    # Time (optional). If missing, we assume uniform sampling per file.
    tx = None
    for cand in ["time","timestamp","t","ms","millis"]:
        if cand in cols:
            tx = cols[cand]
            break

    use_cols = [c for c in [tx, ax, ay, az, gx, gy, gz] if c is not None]
    if len(use_cols) < 6:
        raise ValueError(f"{path}: missing required IMU columns")
    df = df[use_cols].copy()
    df.columns = ["time","ax","ay","az","gx","gy","gz"] if tx else ["ax","ay","az","gx","gy","gz"]
    return df

def resample_to_hz(df, target_hz=50):
    if "time" in df.columns:
        t = df["time"].values.astype(float)
        # If time looks like milliseconds, convert to seconds
        if t.max() > 1e4: t = t / 1000.0
        duration = t[-1] - t[0] if len(t) > 1 else 0
        if duration <= 0:
            # fallback: just return as-is
            return df.drop(columns=["time"]) if "time" in df else df
        new_t = np.arange(t[0], t[-1], 1.0/target_hz)
        out = {"ax":None,"ay":None,"az":None,"gx":None,"gy":None,"gz":None}
        for col in out.keys():
            out[col] = np.interp(new_t, t, df[col].values.astype(float))
        return pd.DataFrame(out)
    else:
        # No timestamps: assume approximately uniform; polyphase resample
        n = len(df)
        # Guess original Hz by assuming ~50 Hz; if wildly different, user should specify
        # For simplicity, use polyphase gain with ratio target/orig ~ 1
        return df.copy()  # keep as-is to avoid distortion without time

def magnitude(x, y, z): return np.sqrt(x*x + y*y + z*z)

def segment_windows(arr, win_len=2.0, hop=1.0, hz=50):
    W = int(win_len * hz)
    H = int(hop * hz)
    if len(arr) < W:  # pad
        pad = np.zeros((W - len(arr), arr.shape[1]))
        arr = np.vstack([arr, pad])
    starts = np.arange(0, max(1, len(arr)-W+1), H, dtype=int)
    return [arr[s:s+W] for s in starts]

def feature_vector(win):
    # win: (T, 6) for ax,ay,az,gx,gy,gz
    a = win[:,0:3]; g = win[:,3:6]
    feats = []

    for M in (a, g):
        # time-domain stats per axis
        feats.extend(M.mean(axis=0))
        feats.extend(M.std(axis=0))
        feats.extend(np.median(M, axis=0))
        feats.extend(np.percentile(M, 10, axis=0))
        feats.extend(np.percentile(M, 90, axis=0))
        # magnitude stats
        mag = magnitude(M[:,0], M[:,1], M[:,2])
        feats += [mag.mean(), mag.std(), np.median(mag),
                  np.percentile(mag,10), np.percentile(mag,90)]
        # zero-crossing rate per axis
        zc = ((M[:-1] * M[1:]) < 0).sum(axis=0) / float(len(M)-1 + 1e-9)
        feats.extend(zc)

        # simple spectral energy around 0–5 Hz, 5–15 Hz bands
        # (with Hann window)
        sig = mag - mag.mean()
        fft = np.fft.rfft(sig * np.hanning(len(sig)))
        freqs = np.fft.rfftfreq(len(sig), d=1.0/50.0)  # assuming 50 Hz
        def band_energy(fmin, fmax):
            idx = np.where((freqs>=fmin) & (freqs<fmax))[0]
            return float((np.abs(fft[idx])**2).sum() / (len(idx)+1e-9))
        feats += [band_energy(0.0,5.0), band_energy(5.0,15.0)]

    # inter-axis correlations for accelerometer and gyro
    for M in (a, g):
        c = np.corrcoef(M.T)
        feats += [c[0,1], c[0,2], c[1,2]]

    return np.array(feats, dtype=np.float32)

def build_dataset(train_dir, target_hz=50, win_len=2.0, hop=1.0):
    X, y, paths = [], [], []
    labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])
    label_to_id = {lab:i for i,lab in enumerate(labels)}
    for lab in labels:
        for f in list_imu_files(os.path.join(train_dir, lab)):
            df = read_imu_csv(f)
            df = resample_to_hz(df, target_hz=target_hz)
            # optional simple "activity VAD": keep only windows with enough variance
            arr = df[["ax","ay","az","gx","gy","gz"]].values.astype(np.float32)
            wins = segment_windows(arr, win_len=win_len, hop=hop, hz=target_hz)
            for w in wins:
                if w.std() < 1e-3:  # skip near-constant windows
                    continue
                X.append(feature_vector(w))
                y.append(label_to_id[lab])
                paths.append(f)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    return X, y, paths, labels

def train(train_dir="data/train", model_out="models/imu_rf.joblib",
          labels_out="models/labels.json", target_hz=50, win_len=2.0, hop=1.0,
          model="rf"):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    X, y, _, labels = build_dataset(train_dir, target_hz, win_len, hop)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    if model == "knn":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ])
    else:
        # Random Forest (no scaling needed)
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        )

    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xva)
    print("\nValidation report:")
    print(classification_report(yva, yhat, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yva, yhat))

    dump(clf, model_out)
    with open(labels_out, "w") as f:
        json.dump({"labels": labels}, f, indent=2)
    print(f"\nSaved model -> {model_out}\nSaved labels -> {labels_out}")

def load_model(model_path, labels_path):
    clf = load(model_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)["labels"]
    return clf, labels

def predict_folder(folder="data/test", model_path="models/imu_rf.joblib",
                   labels_path="models/labels.json", target_hz=50,
                   win_len=2.0, hop=1.0, min_conf=0.0):
    clf, labels = load_model(model_path, labels_path)
    files = list_imu_files(folder)
    if not files:
        print(f"No IMU files in {folder}")
        return
    print(f"\nScoring {len(files)} file(s) in '{folder}'")
    for f in files:
        df = read_imu_csv(f)
        df = resample_to_hz(df, target_hz=target_hz)
        arr = df[["ax","ay","az","gx","gy","gz"]].values.astype(np.float32)
        wins = segment_windows(arr, win_len=win_len, hop=hop, hz=target_hz)
        if not wins:
            print(os.path.basename(f), "-> no windows")
            continue
        X = np.vstack([feature_vector(w) for w in wins])
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X).mean(axis=0)
            top = int(np.argmax(probs))
            conf = float(probs[top])
            pred = labels[top] if conf >= min_conf else "unknown"
            print(f"{os.path.basename(f):<32} -> {pred}  (conf={conf:.3f})")
        else:
            preds = clf.predict(X)
            # majority vote
            top = np.bincount(preds).argmax()
            print(f"{os.path.basename(f):<32} -> {labels[top]}")

def main():
    ap = argparse.ArgumentParser(description="IMU-based human movement classification")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--train_dir", default="data/train")
    tr.add_argument("--model", default="models/imu_rf.joblib")
    tr.add_argument("--labels", default="models/labels.json")
    tr.add_argument("--target_hz", type=int, default=50)
    tr.add_argument("--win_len", type=float, default=2.0)
    tr.add_argument("--hop", type=float, default=1.0)
    tr.add_argument("--algo", choices=["rf","knn"], default="rf")

    pr = sub.add_parser("predict")
    pr.add_argument("--folder", default="data/test")
    pr.add_argument("--model", default="models/imu_rf.joblib")
    pr.add_argument("--labels", default="models/labels.json")
    pr.add_argument("--target_hz", type=int, default=50)
    pr.add_argument("--win_len", type=float, default=2.0)
    pr.add_argument("--hop", type=float, default=1.0)
    pr.add_argument("--min_conf", type=float, default=0.0)

    args = ap.parse_args()
    if args.cmd == "train":
        train(train_dir=args.train_dir, model_out=args.model, labels_out=args.labels,
              target_hz=args.target_hz, win_len=args.win_len, hop=args.hop, model=args.algo)
    else:
        predict_folder(folder=args.folder, model_path=args.model, labels_path=args.labels,
                       target_hz=args.target_hz, win_len=args.win_len, hop=args.hop, min_conf=args.min_conf)

if __name__ == "__main__":
    main()