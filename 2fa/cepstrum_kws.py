#!/usr/bin/env python3
"""
Cepstrum-based keyword + speaker verifier.

Directory layout (relative to where you run this):
  data/pos/   -> positive examples (target person saying the target word)
  data/neg/   -> negative examples (anything else)
  testing/    -> audio files to classify

Usage:
  python cepstrum_kws.py train
  python cepstrum_kws.py predict testing
  python cepstrum_kws.py eval testing    # prints per-file prediction, overall metrics

Models are saved to models/cepstrum_model.json (+ .npy blobs).
"""

import os, json, sys, math, glob, hashlib, warnings
from dataclasses import dataclass, asdict
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, lfilter, get_window

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_wav_any_mono(path, target_sr=16000):
    """Reads wav/pcm; if stereo, averages channels; resamples if needed."""
    sr, x = wavfile.read(path)
    # normalize to float32 [-1,1]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)

    if x.ndim == 2:
        x = x.mean(axis=1)

    if sr != target_sr:
        n = int(round(len(x) * target_sr / sr))
        if n <= 0: n = 1
        x = resample(x, n)
        sr = target_sr

    # trim leading/trailing tiny DC
    if len(x) == 0:
        return sr, x
    x = x - np.mean(x)
    return sr, x

def frame_signal(x, sr, frame_ms=25.0, hop_ms=10.0, window='hann'):
    N = int(sr * frame_ms / 1000.0)
    H = int(sr * hop_ms / 1000.0)
    if N < 8: N = 8
    if H < 1: H = 1
    w = get_window(window, N, fftbins=True).astype(np.float32)
    T = 1 + (len(x) - N) // H if len(x) >= N else 0
    if T <= 0:
        return np.zeros((0, N), dtype=np.float32), N, H, w
    frames = np.stack([x[i*H:i*H+N]*w for i in range(T)]).astype(np.float32)
    return frames, N, H, w

def simple_vad(frames, energy_thresh_db=-40.0, zcr_max=0.25):
    """Return boolean mask of voiced frames based on log energy + zero-crossing."""
    if len(frames) == 0:
        return np.zeros((0,), dtype=bool)
    # log energy per frame
    eps = 1e-10
    E = 10.0*np.log10(np.maximum(np.sum(frames**2, axis=1), eps))
    # normalize energies relative to per-utterance max
    E_rel = E - np.max(E)
    # zero crossing rate
    signs = np.sign(frames)
    signs[signs==0] = 1
    zc = np.mean((signs[:,1:] != signs[:,:-1]).astype(np.float32), axis=1)
    vad = (E_rel >= energy_thresh_db) & (zc <= zcr_max)
    # smooth a bit: keep isolated holes out
    if vad.size >= 3:
        pad = np.pad(vad.astype(np.int32), (1,1))
        run = (pad[:-2] + pad[1:-1] + pad[2:])
        vad = (run >= 2)
    return vad

def real_cepstrum(frames, n_ceps=20):
    """
    Real cepstrum per frame:
      c = IFFT(log(|FFT(frame)| + eps))
    Return first n_ceps coefficients (excluding c0 by default in many pipelines,
    but we INCLUDE c0 here and let the model learn its usefulness).
    """
    if len(frames) == 0:
        return np.zeros((0, n_ceps), dtype=np.float32)
    # Next power of two FFT for speed/stability
    N = frames.shape[1]
    nfft = 1 << (N-1).bit_length()
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    mag = np.abs(spec) + 1e-8
    log_mag = np.log(mag)
    cep = np.fft.irfft(log_mag, n=nfft, axis=1).real
    # Take first n_ceps (low-quefrency envelope)
    cep = cep[:, :n_ceps].astype(np.float32)
    # liftering (gentle) to balance coefficients
    lifter = 1.0 + 0.6*np.arange(cep.shape[1], dtype=np.float32)
    cep *= lifter
    return cep

def zscore(x, mean, std, eps=1e-6):
    return (x - mean) / (std + eps)

def dtw_distance(A, B):
    """
    Basic DTW (Euclidean). A: T1xd, B: T2xd. Returns scalar distance.
    """
    if len(A)==0 or len(B)==0:
        return float('inf')
    T1, d = A.shape
    T2, _ = B.shape
    D = np.full((T1+1, T2+1), np.inf, dtype=np.float64)
    D[0,0] = 0.0
    # Precompute per-step cost
    # Use vectorized broadcast for speed in chunks
    for i in range(1, T1+1):
        ai = A[i-1]
        # compute distances to all bj
        diff = B - ai  # (T2,d)
        row_cost = np.sqrt(np.sum(diff*diff, axis=1))  # (T2,)
        for j in range(1, T2+1):
            D[i,j] = row_cost[j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[T1,T2] / (T1+T2))  # normalized a bit by path length

def resample_time_sequence(X, T=50):
    """Resample a Timestep x Dim sequence to exactly T steps (linear along time)."""
    if len(X) == 0:
        return np.zeros((T, X.shape[1] if X.ndim==2 else 20), dtype=np.float32)
    if len(X) == T:
        return X.astype(np.float32)
    t_old = np.linspace(0.0, 1.0, num=len(X), endpoint=True)
    t_new = np.linspace(0.0, 1.0, num=T, endpoint=True)
    D = X.shape[1]
    Y = np.zeros((T, D), dtype=np.float32)
    for k in range(D):
        Y[:,k] = np.interp(t_new, t_old, X[:,k])
    return Y

def hash_path(path):
    return hashlib.sha1(path.encode('utf-8')).hexdigest()[:10]

# -----------------------------
# Model containers
# -----------------------------

@dataclass
class CepstrumModel:
    sr: int
    n_ceps: int
    frame_ms: float
    hop_ms: float
    template_T: int
    # Speaker Gaussian (diagonal)
    spk_mean: list
    spk_std: list
    # Keyword template trajectory (T x n_ceps)
    kw_template_path: str
    # Score normalization (from train pos/neg)
    speaker_mu: float
    speaker_sigma: float
    dtw_mu: float
    dtw_sigma: float
    # Fusion + threshold
    alpha: float          # weight for speaker z
    beta: float           # weight for -dtw z
    decision_threshold: float

    def save(self, model_dir="models", name="cepstrum_model"):
        ensure_dir(model_dir)
        jpath = os.path.join(model_dir, f"{name}.json")
        tpath = os.path.join(model_dir, f"{name}_kw_template.npy")
        with open(jpath, "w") as f:
            d = asdict(self)
            d['kw_template_path'] = tpath  # ensure stored relative path
            json.dump(d, f, indent=2)
        # Save the keyword template trajectory
        np.save(tpath, np.load(self.kw_template_path))

    @staticmethod
    def load(model_dir="models", name="cepstrum_model"):
        jpath = os.path.join(model_dir, f"{name}.json")
        if not os.path.isfile(jpath):
            raise FileNotFoundError(f"Missing model JSON: {jpath}")
        with open(jpath, "r") as f:
            d = json.load(f)
        m = CepstrumModel(**d)
        # Make sure template path exists
        if not os.path.isfile(m.kw_template_path):
            raise FileNotFoundError(f"Missing template .npy: {m.kw_template_path}")
        return m

# -----------------------------
# Feature extraction pipeline
# -----------------------------

def extract_voiced_cepstra(path, sr=16000, n_ceps=20, frame_ms=25.0, hop_ms=10.0):
    sr, x = read_wav_any_mono(path, target_sr=sr)
    if len(x) == 0:
        return np.zeros((0, n_ceps), dtype=np.float32)
    # pre-emphasis (mild)
    x = lfilter([1.0, -0.97], [1.0], x).astype(np.float32)
    frames, N, H, win = frame_signal(x, sr, frame_ms=frame_ms, hop_ms=hop_ms, window='hann')
    vad = simple_vad(frames, energy_thresh_db=-40.0, zcr_max=0.25)
    voiced = frames[vad]
    return real_cepstrum(voiced, n_ceps=n_ceps)

# -----------------------------
# Training
# -----------------------------

def train_model(pos_dir="data/pos", neg_dir="data/neg",
                sr=16000, n_ceps=20, frame_ms=25.0, hop_ms=10.0, template_T=50):
    pos_files = sorted(glob.glob(os.path.join(pos_dir, "*.wav")))
    neg_files = sorted(glob.glob(os.path.join(neg_dir, "*.wav")))
    if len(pos_files) == 0:
        raise RuntimeError(f"No WAV files found in {pos_dir}")
    if len(neg_files) == 0:
        warnings.warn(f"No WAV files found in {neg_dir}; thresholds may be poor.")

    # 1) Build speaker model (Gaussian on cepstral coefficients across all pos voiced frames)
    spk_ceps = []
    pos_seq_fixed = []  # resampled sequences for keyword template
    for p in pos_files:
        C = extract_voiced_cepstra(p, sr=sr, n_ceps=n_ceps, frame_ms=frame_ms, hop_ms=hop_ms)
        if len(C) == 0:
            continue
        spk_ceps.append(C)
        pos_seq_fixed.append(resample_time_sequence(C, T=template_T))
    if len(spk_ceps) == 0:
        raise RuntimeError("No voiced frames found in positive data.")
    spk_stack = np.concatenate(spk_ceps, axis=0)  # (total_frames, n_ceps)
    spk_mean = np.mean(spk_stack, axis=0)
    spk_std  = np.std(spk_stack, axis=0) + 1e-6

    # 2) Build keyword template trajectory by averaging time-normalized sequences
    kw_template = np.mean(np.stack(pos_seq_fixed, axis=0), axis=0).astype(np.float32)  # (T, n_ceps)

    # 3) Collect score distributions on train for normalization & fusion tuning
    def speaker_score(C):
        # z-scored frame-wise distance to speaker Gaussian mean; use average -Mahalanobis-like
        if len(C)==0:
            return +np.inf
        Z = zscore(C, spk_mean, spk_std)
        d = np.mean(np.sqrt(np.sum(Z**2, axis=1)))
        # Smaller is better → convert to similarity-like by negation
        return -d

    def keyword_score(C):
        if len(C)==0:
            return +np.inf
        R = resample_time_sequence(C, T=template_T)
        return -dtw_distance(R, kw_template)  # larger is better (negative distance)

    pos_spk = []
    pos_kw  = []
    for p in pos_files:
        C = extract_voiced_cepstra(p, sr=sr, n_ceps=n_ceps, frame_ms=frame_ms, hop_ms=hop_ms)
        pos_spk.append(speaker_score(C))
        pos_kw.append(keyword_score(C))

    neg_spk = []
    neg_kw  = []
    for n in neg_files:
        C = extract_voiced_cepstra(n, sr=sr, n_ceps=n_ceps, frame_ms=frame_ms, hop_ms=hop_ms)
        neg_spk.append(speaker_score(C))
        neg_kw.append(keyword_score(C))

    # z-normalize each score stream using both pos+neg (robust-ish)
    all_spk = np.array(pos_spk + neg_spk, dtype=np.float64)
    all_kw  = np.array(pos_kw  + neg_kw , dtype=np.float64)
    spk_mu, spk_sigma = float(np.mean(all_spk)), float(np.std(all_spk) + 1e-6)
    kw_mu,  kw_sigma  = float(np.mean(all_kw)),  float(np.std(all_kw)  + 1e-6)

    def fuse(s_spk, s_kw, alpha=0.5, beta=0.5):
        z1 = (s_spk - spk_mu)/spk_sigma
        z2 = (s_kw  -  kw_mu)/kw_sigma
        # Final score: alpha * speaker_z  +  beta * keyword_z
        return alpha*z1 + beta*z2

    # grid-search tiny fusion weights, pick threshold maximizing F1 on train
    if len(neg_files) == 0:
        alpha, beta = 0.5, 0.5
        thr = np.median([(fuse(s1,s2,alpha,beta)) for s1,s2 in zip(pos_spk,pos_kw)])
    else:
        best = (-1.0, 0.5, 0.5, 0.0) # (F1, alpha, beta, thr)
        for alpha in np.linspace(0.2,0.8,7):
            beta = 1.0 - alpha
            scores = [fuse(s1,s2,alpha,beta) for s1,s2 in zip(pos_spk,pos_kw)]
            labels = [1]*len(scores)
            nscores = [fuse(s1,s2,alpha,beta) for s1,s2 in zip(neg_spk,neg_kw)]
            scores += nscores
            labels += [0]*len(nscores)
            # candidate thresholds: midpoints between sorted scores
            order = np.argsort(scores)
            ss = np.array(scores)[order]
            ll = np.array(labels)[order]
            # evaluate thresholds
            for k in range(len(ss)):
                thr = ss[k]
                pred = (np.array(scores) >= thr).astype(int)
                tp = int(np.sum((pred==1) & (np.array(labels)==1)))
                fp = int(np.sum((pred==1) & (np.array(labels)==0)))
                fn = int(np.sum((pred==0) & (np.array(labels)==1)))
                prec = tp/(tp+fp+1e-9)
                rec  = tp/(tp+fn+1e-9)
                f1   = 2*prec*rec/(prec+rec+1e-9)
                if f1 > best[0]:
                    best = (f1, alpha, beta, float(thr))
        _, alpha, beta, thr = best

    # Save template npy first
    ensure_dir("models")
    tpath = os.path.join("models", "cepstrum_model_kw_template.npy")
    np.save(tpath, kw_template)

    model = CepstrumModel(
        sr=sr, n_ceps=n_ceps, frame_ms=frame_ms, hop_ms=hop_ms, template_T=template_T,
        spk_mean=spk_mean.tolist(), spk_std=spk_std.tolist(),
        kw_template_path=tpath,
        speaker_mu=spk_mu, speaker_sigma=spk_sigma,
        dtw_mu=kw_mu, dtw_sigma=kw_sigma,
        alpha=float(alpha), beta=float(beta),
        decision_threshold=float(thr),
    )
    model.save()
    print("[train] saved model to models/cepstrum_model.json")
    print(f"[train] fusion alpha={alpha:.3f} beta={beta:.3f} threshold={thr:.3f}")
    print(f"[train] #pos={len(pos_files)} #neg={len(neg_files)}")

# -----------------------------
# Inference
# -----------------------------

def load_model():
    return CepstrumModel.load()

def score_file(path, model: CepstrumModel):
    C = extract_voiced_cepstra(path, sr=model.sr, n_ceps=model.n_ceps,
                               frame_ms=model.frame_ms, hop_ms=model.hop_ms)
    spk_mean = np.array(model.spk_mean, dtype=np.float32)
    spk_std  = np.array(model.spk_std,  dtype=np.float32)

    if len(C)==0:
        speaker_sim = -np.inf
        kw_sim = -np.inf
    else:
        Z = zscore(C, spk_mean, spk_std)
        speaker_sim = -float(np.mean(np.sqrt(np.sum(Z**2, axis=1))))
        kw_template = np.load(model.kw_template_path)
        R = resample_time_sequence(C, T=model.template_T)
        kw_sim = -float(dtw_distance(R, kw_template))

    z_spk = (speaker_sim - model.speaker_mu)/(model.speaker_sigma + 1e-6)
    z_kw  = (kw_sim      - model.dtw_mu     )/(model.dtw_sigma  + 1e-6)
    fused = model.alpha * z_spk + model.beta * z_kw
    pred  = int(fused >= model.decision_threshold)
    debug = {
        "speaker_sim": speaker_sim, "kw_sim": kw_sim,
        "z_spk": z_spk, "z_kw": z_kw, "fused": fused,
        "thr": model.decision_threshold
    }
    return pred, debug

def predict_folder(folder):
    model = load_model()
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if len(files)==0:
        print(f"[predict] No WAV files in {folder}")
        return
    for f in files:
        y, dbg = score_file(f, model)
        lab = "POSITIVE" if y==1 else "NEGATIVE"
        print(f"{os.path.basename(f)}\t{lab}\t(fused={dbg['fused']:.3f}, thr={dbg['thr']:.3f})")

def eval_folder(folder):
    model = load_model()
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if len(files)==0:
        print(f"[eval] No WAV files in {folder}")
        return
    y_true, y_pred = [], []
    for f in files:
        # Heuristic: treat any filename containing 'pos' as positive label in testing (optional)
        # If you have testing labels elsewhere, adapt here.
        true = 1 if ("pos" in os.path.basename(f).lower()) else 0
        pred, _ = score_file(f, model)
        y_true.append(true)
        y_pred.append(pred)
        print(f"{os.path.basename(f)}\tPRED={'POS' if pred==1 else 'NEG'}\tTRUE={'POS' if true==1 else 'NEG'}")
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = int(np.sum((y_pred==1)&(y_true==1)))
    tn = int(np.sum((y_pred==0)&(y_true==0)))
    fp = int(np.sum((y_pred==1)&(y_true==0)))
    fn = int(np.sum((y_pred==0)&(y_true==1)))
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    acc = (tp+tn)/max(1,len(y_true))
    print(f"\n[eval] acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} (tp={tp} fp={fp} fn={fn} tn={tn})")

# -----------------------------
# CLI
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "train":
        train_model()
    elif cmd == "predict":
        folder = sys.argv[2] if len(sys.argv) >= 3 else "testing"
        predict_folder(folder)
    elif cmd == "eval":
        folder = sys.argv[2] if len(sys.argv) >= 3 else "testing"
        eval_folder(folder)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(2)

if __name__ == "__main__":
    main()
