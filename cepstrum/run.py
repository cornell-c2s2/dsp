import os
import joblib
import numpy as np
import librosa

# Use the same MFCC configuration as during training
N_MFCC = 20
MODEL_PATH = "scrubjay_svm.joblib"
AUDIO_FOLDER = "full-sr" 

def mfcc_stats(y, sr):
    """Extract mean and std of MFCC coefficients."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

def load_audio_features(audio_path, sr=None):
    """Load audio and extract MFCC stats."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        return mfcc_stats(y, sr)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    # Load model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}\n")

    # Get all audio files in the folder
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    if not audio_files:
        print(f"No audio files found in {AUDIO_FOLDER}")
        return

    for filename in sorted(audio_files):
        path = os.path.join(AUDIO_FOLDER, filename)
        features = load_audio_features(path)

        if features is None:
            continue

        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        label = "Scrub Jay Call ✅" if prediction == 1 else "Not Scrub Jay ❌"
        print(f"{filename:30s} → {label} (confidence: {prob[prediction]:.2f})")

if __name__ == "__main__":
    main()
