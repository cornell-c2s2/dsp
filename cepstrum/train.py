import os, glob, random, joblib, numpy as np
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ---------- hyper‑params you will tweak ----------
N_MFCC       = 20          # 13–40 common; keep small for tiny datasets
SAMPLE_RATE  = None        # None = use file’s native sr
AUGMENT_PROB = 0.5         # chance of applying a random augmentation
MODEL_PATH   = "scrubjay_svm.joblib"
# -------------------------------------------------

def extract_mfcc(path, n_mfcc=N_MFCC, sr=SAMPLE_RATE):
    """Return a 2×N_MFCC vector (mean & std of each coeff)."""
    y, sr = librosa.load(path, sr=sr)
    # quick‑n‑dirty filtering: discard clips <0.25 s
    if y.shape[0] < sr // 4:
        raise ValueError("Clip too short")
    # basic augmentation (only on training set later)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def augment(y, sr):
    """Very small dataset?  Do simple time‑stretch / pitch‑shift / noise."""
    choice = random.choice(["stretch", "pitch", "noise"])
    if choice == "stretch":
        rate = random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate)
    elif choice == "pitch":
        steps = random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y, sr, n_steps=steps)
    else:
        y = y + 0.005 * np.random.randn(len(y))
    return y

def mfcc_stats(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

def load_dataset(root="data"):
    X, y = [], []
    for label, sub in [(1, "pos"), (0, "neg")]:
        for fp in glob.glob(os.path.join(root, sub, "*")):
            try:
                y_audio, sr = librosa.load(fp, sr=SAMPLE_RATE)
                feats = mfcc_stats(y_audio, sr)
                X.append(feats)
                y.append(label)

                # crude augmentation to enlarge minority class
                if label == 1 and random.random() < AUGMENT_PROB:
                    feats_aug = mfcc_stats(augment(y_audio, sr), sr)
                    X.append(feats_aug)
                    y.append(label)

            except Exception as e:
                print(f"Skipped {fp}: {e}")
    return np.vstack(X), np.array(y)

def main():
    X, y = load_dataset()
    print("Dataset size:", X.shape)

    # 80/20 split for a quick sanity check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # A very small dataset often likes a simple model
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, class_weight="balanced", C=10, gamma="scale")
    )
    clf.fit(X_train, y_train)

    print("\nHeld‑out test set:")
    print(classification_report(y_test, clf.predict(X_test), digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, clf.predict(X_test)))

    # Optional k‑fold cross‑val (good when data are scarce)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
    print("Stratified 5‑fold F1:", scores.mean(), "+/-", scores.std())

    # Persist the whole pipeline (scaler + SVM)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model ➜ {MODEL_PATH}")

if __name__ == "__main__":
    main()
