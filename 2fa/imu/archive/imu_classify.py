
# imu_classify.py (updated for this dataset & per-row predictions)
# Usage:
#   Train (RF):   python imu_classify.py train --train_dir data/train --algo rf --hz 50
#   Train (kNN):  python imu_classify.py train --train_dir data/train --algo knn --hz 50
#   Predict files:python imu_classify.py predict --folder data/test --hz 50
#   Predict rows: python imu_classify.py predict_rows --file data/test/Sub9_clapping.csv --row_window 64 --hz 50 --out predictions/Sub9_clapping_rows.csv
#
# Notes
# - Works with CSVs produced by convert_xlsx_to_csv.py.
#   • If you exported a single group (e.g., ARM): expected columns -> ax,ay,az,gx,gy,gz
#   • If you exported ALL groups: expected columns -> ax_arm,...,gz_arm, ax_leg,...,gz_leg, ax_neck,...,gz_neck
# - If no timestamps, we assume uniform sampling at --hz (default 50 Hz).

import os, glob, json, re, argparse, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from joblib import dump, load

warnings.filterwarnings("ignore", category=UserWarning)
IMU_EXTS = (".csv", ".tsv")
AXES = ["ax","ay","az","gx","gy","gz"]


def list_imu_files(folder: str) -> List[str]:
    files = []
    for ext in IMU_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def read_imu_csv(path: str) -> pd.DataFrame:
    sep = "," if path.lower().endswith(".csv") else "	"
    df = pd.read_csv(path, sep=sep)
    # Normalize header to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Case 1: single-group format ax..gz present
    if all(c in df.columns for c in AXES):
        return df[AXES].copy()

    # Case 2: multi-group format with suffixes _arm/_leg/_neck
    # Build in order ARM, LEG, NECK if present
    groups = ["arm","leg","neck"]
    cols = []
    for G in groups:
        gcols = [f"{a}_{G}" for a in AXES]
        if all(gc in df.columns for gc in gcols):
            cols.extend(gcols)
    if cols:
        return df[cols].copy()

    # Attempt to recover columns with other suffixes
    suffixes = sorted(set(m.group(1) for m in (re.match(r"^(ax|ay|az|gx|gy|gz)_(.+)$", c) for c in df.columns) if m))
    if suffixes:
        chosen = []
        for s in suffixes:
            gcols = [f"{a}_{s}" for a in AXES]
            if all(gc in df.columns for gc in gcols):
                chosen.extend(gcols)
        if chosen:
            return df[chosen].copy()

    raise ValueError(f"{path}: could not find IMU columns (ax..gz or ax_* suffixes)")


def magnitude(x, y, z):
    return np.sqrt(x*x + y*y + z*z)


def segment_windows(arr: np.ndarray, win_len: float = 2.0, hop: float = 1.0, hz: int = 50) -> List[np.ndarray]:
    W = int(max(1, round(win_len * hz)))
    H = int(max(1, round(hop * hz)))
    if len(arr) < W:
        pad = np.zeros((W - len(arr), arr.shape[1]), dtype=arr.dtype)
        arr = np.vstack([arr, pad])
    starts = np.arange(0, max(1, len(arr) - W + 1), H, dtype=int)
    return [arr[s:s+W] for s in starts]


def feature_vector_multi(win: np.ndarray, hz: int = 50) -> np.ndarray:
    """Compute features over possibly multiple sensor groups.
    win shape: (T, 6*G), group blocks are [ax,ay,az,gx,gy,gz] for each group in order.
    """
    T, D = win.shape
    G = D // 6
    feats_all = []

    for g in range(G):
        block = win[:, g*6:(g+1)*6]
        a = block[:, 0:3]
        gy = block[:, 3:6]

        # time stats per axis
        for M in (a, gy):
            feats_all.extend(M.mean(axis=0))
            feats_all.extend(M.std(axis=0))
            feats_all.extend(np.median(M, axis=0))
            feats_all.extend(np.percentile(M, 10, axis=0))
            feats_all.extend(np.percentile(M, 90, axis=0))
            # magnitude stats
            mag = magnitude(M[:,0], M[:,1], M[:,2])
            feats_all += [mag.mean(), mag.std(), np.median(mag),
                          np.percentile(mag,10), np.percentile(mag,90)]
            # zero-cross rate per axis
            if len(M) > 1:
                zc = ((M[:-1] * M[1:]) < 0).sum(axis=0) / float(len(M)-1)
            else:
                zc = np.zeros(3)
            feats_all.extend(zc)
            # simple spectral energies (0–5 Hz, 5–15 Hz) from magnitude
            sig = mag - mag.mean()
            fft = np.fft.rfft(sig * np.hanning(len(sig)))
            freqs = np.fft.rfftfreq(len(sig), d=1.0/float(hz))
            def band_energy(fmin, fmax):
                idx = np.where((freqs>=fmin) & (freqs<fmax))[0]
                return float((np.abs(fft[idx])**2).sum() / (len(idx)+1e-9))
            feats_all += [band_energy(0.0,5.0), band_energy(5.0,15.0)]
        # inter-axis correlations for acc and gyro
        for M in (a, gy):
            if M.shape[0] >= 3:
                c = np.corrcoef(M.T)
                feats_all += [c[0,1], c[0,2], c[1,2]]
            else:
                feats_all += [0.0, 0.0, 0.0]

    return np.array(feats_all, dtype=np.float32)


def build_dataset(train_dir: str, hz: int = 50, win_len: float = 2.0, hop: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    X, y, paths = [], [], []
    labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    label_to_id = {lab:i for i,lab in enumerate(labels)}

    for lab in labels:
        for f in list_imu_files(os.path.join(train_dir, lab)):
            df = read_imu_csv(f)
            arr = df.values.astype(np.float32)
            wins = segment_windows(arr, win_len=win_len, hop=hop, hz=hz)
            for w in wins:
                if w.std() < 1e-6:
                    continue
                X.append(feature_vector_multi(w, hz=hz))
                y.append(label_to_id[lab])
                paths.append(f)
    if not X:
        raise RuntimeError("No training windows found. Check your data/train layout and CSV columns.")
    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    return X, y, paths, labels


def train(train_dir="data/train", model_out="models/imu_rf.joblib", labels_out="models/labels.json",
          hz: int = 50, win_len: float = 2.0, hop: float = 1.0, algo: str = "rf"):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    X, y, _, labels = build_dataset(train_dir, hz=hz, win_len=win_len, hop=hop)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    if algo == "knn":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ])
    else:
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        )

    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xva)
    print("Validation report:")
    print(classification_report(yva, yhat, digits=4))
    print("Confusion matrix:", confusion_matrix(yva, yhat))

    dump(clf, model_out)
    with open(labels_out, "w") as f:
        json.dump({"labels": labels}, f, indent=2)
    print(f"Saved model -> {model_out} Saved labels -> {labels_out}")


def load_model(model_path: str, labels_path: str):
    clf = load(model_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)["labels"]
    return clf, labels


def predict_folder(folder="data/test", model_path="models/imu_rf.joblib", labels_path="models/labels.json",
                   hz: int = 50, win_len: float = 2.0, hop: float = 1.0, min_conf: float = 0.0):
    clf, labels = load_model(model_path, labels_path)
    files = list_imu_files(folder)
    if not files:
        print(f"No IMU files in {folder}")
        return
    print(f"Scoring {len(files)} file(s) in '{folder}' (windowed majority)")
    for f in files:
        df = read_imu_csv(f)
        arr = df.values.astype(np.float32)
        wins = segment_windows(arr, win_len=win_len, hop=hop, hz=hz)
        if not wins:
            print(os.path.basename(f), "-> no windows")
            continue
        X = np.vstack([feature_vector_multi(w, hz=hz) for w in wins])
        if hasattr(clf, "predict_proba"):
            probs_win = clf.predict_proba(X)
            probs = probs_win.mean(axis=0)
            top = int(np.argmax(probs))
            conf = float(probs[top])
            pred = labels[top] if conf >= min_conf else "unknown"
            print(f"{os.path.basename(f):<40} -> {pred}  (conf={conf:.3f})")
        else:
            preds = clf.predict(X)
            top = int(np.bincount(preds).argmax())
            print(f"{os.path.basename(f):<40} -> {labels[top]}")


def predict_rows(file: str, model_path="models/imu_rf.joblib", labels_path="models/labels.json",
                 hz: int = 50, row_window: int = 64, min_conf: float = 0.0, out: str = None):
    """Per-row predictions: for each sample (row) in the CSV, build a centered window of length
    `row_window` samples, extract features, and predict a label; write results to CSV."""
    clf, labels = load_model(model_path, labels_path)
    df = read_imu_csv(file)
    arr = df.values.astype(np.float32)
    n, d = arr.shape
    W = max(1, int(row_window))
    half = W // 2

    preds, confs = [], []
    if hasattr(clf, "predict_proba"):
        for i in range(n):
            s = max(0, i - half)
            e = min(n, s + W)
            s = max(0, e - W)
            win = arr[s:e]
            if len(win) < W:
                pad = np.zeros((W - len(win), d), dtype=arr.dtype)
                win = np.vstack([win, pad])
            x = feature_vector_multi(win, hz=hz).reshape(1, -1)
            p = clf.predict_proba(x)[0]
            k = int(np.argmax(p))
            preds.append(labels[k] if float(p[k]) >= min_conf else "unknown")
            confs.append(float(p[k]))
    else:
        for i in range(n):
            s = max(0, i - half)
            e = min(n, s + W)
            s = max(0, e - W)
            win = arr[s:e]
            if len(win) < W:
                pad = np.zeros((W - len(win), d), dtype=arr.dtype)
                win = np.vstack([win, pad])
            x = feature_vector_multi(win, hz=hz).reshape(1, -1)
            k = int(clf.predict(x)[0])
            preds.append(labels[k])
            confs.append(np.nan)

    out_df = pd.DataFrame({"row_index": np.arange(n), "pred": preds, "conf": confs})
    if out is None:
        out = os.path.splitext(file)[0] + "_rowpreds.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
    out_df.to_csv(out, index=False)
    print(f"Wrote per-row predictions -> {out}")


def main():
    ap = argparse.ArgumentParser(description="IMU-based movement classification (Excel pipeline)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--train_dir", default="data/train")
    tr.add_argument("--model", default="models/imu_rf.joblib")
    tr.add_argument("--labels", default="models/labels.json")
    tr.add_argument("--hz", type=int, default=50)
    tr.add_argument("--win_len", type=float, default=2.0)
    tr.add_argument("--hop", type=float, default=1.0)
    tr.add_argument("--algo", choices=["rf","knn"], default="rf")

    pr = sub.add_parser("predict")
    pr.add_argument("--folder", default="data/test")
    pr.add_argument("--model", default="models/imu_rf.joblib")
    pr.add_argument("--labels", default="models/labels.json")
    pr.add_argument("--hz", type=int, default=50)
    pr.add_argument("--win_len", type=float, default=2.0)
    pr.add_argument("--hop", type=float, default=1.0)
    pr.add_argument("--min_conf", type=float, default=0.0)

    prr = sub.add_parser("predict_rows")
    prr.add_argument("--file", required=True)
    prr.add_argument("--model", default="models/imu_rf.joblib")
    prr.add_argument("--labels", default="models/labels.json")
    prr.add_argument("--hz", type=int, default=50)
    prr.add_argument("--row_window", type=int, default=64)
    prr.add_argument("--min_conf", type=float, default=0.0)
    prr.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.cmd == "train":
        train(train_dir=args.train_dir, model_out=args.model, labels_out=args.labels,
              hz=args.hz, win_len=args.win_len, hop=args.hop, algo=args.algo)
    elif args.cmd == "predict":
        predict_folder(folder=args.folder, model_path=args.model, labels_path=args.labels,
                       hz=args.hz, win_len=args.win_len, hop=args.hop, min_conf=args.min_conf)
    else:
        predict_rows(file=args.file, model_path=args.model, labels_path=args.labels,
                     hz=args.hz, row_window=args.row_window, min_conf=args.min_conf, out=args.out)

if __name__ == "__main__":
    main()
