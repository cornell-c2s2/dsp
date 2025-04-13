import joblib
import numpy as np
import librosa

# Use the same MFCC hyper-parameter as during training
N_MFCC = 20

def mfcc_stats(y, sr):
    """Compute mean and standard deviation statistics of MFCC coefficients."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # Concatenate the mean and std of each MFCC coefficient into one feature vector.
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

def load_audio_features(audio_path, sr=None):
    """Load an audio file, and return a feature vector computed from its MFCCs."""
    try:
        # Load the audio file. If sr is None, use the file's native sampling rate.
        y, sr = librosa.load(audio_path, sr=sr)
        # Compute and return MFCC statistics.
        return mfcc_stats(y, sr)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    # Path to the saved model from training (adjust if needed)
    model_path = "scrubjay_svm.joblib"
    # Load your trained pipeline (scaler + classifier)
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Path to the new audio file
    new_audio_path = "full-sr/1363v2-sj.WAV"
    #new_audio_path = "full-sr/1060-control.wav"
    #new_audio_path = "full-sr/1809v2-not-sj.wav"
    features = load_audio_features(new_audio_path)

    if features is None:
        print("Error processing the audio file.")
        return

    # Reshape features to a 2D array (1 sample x number of features)
    features = features.reshape(1, -1)

    # Get prediction. Model is trained with labels: 1 (Florida Scrub Jay call), 0 (other)
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)  # optional: probabilities of each class

    # Display the results
    class_name = "Scrub Jay Call" if prediction[0] == 1 else "No Scrub Jay Call"
    print(f"Prediction: {class_name}")
    print("Prediction probabilities:", probabilities)

if __name__ == "__main__":
    main()
