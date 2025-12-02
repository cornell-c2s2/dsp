from keyword_classifier import AudioClassifier

clf = AudioClassifier(n_mfcc=13, max_length=1000)
clf.load_model("models/audio_classifier_model.keras")

pos_path = "../../data/testing/stop_121417.wav"      # contains "stop"
neg_path = "../../data/testing/bed__common_voice_en_82827.wav"  # does NOT contain "stop"

for path, name in [(pos_path, "POS"), (neg_path, "NEG")]:
    mfcc = clf.extract_mfcc(path)
    mfcc_scaled = clf.scaler.transform(mfcc.reshape(1, -1))
    prob = clf.model.predict(mfcc_scaled)[0, 0]
    print(f"{name} clip:", path)
    print("  Python probability:", prob)
