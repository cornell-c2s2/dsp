import argparse
import json
from speaker.gmm_utils import load_model, predict_file
from keyword_classifier import AudioClassifier

DEFAULT_TARGET = "models/target_gmm.joblib"
DEFAULT_UBM = "models/ubm_gmm.joblib"
DEFAULT_KW_MODEL = "models/audio_classifier_model.keras"

DEFAULT_SPEAKER_THRESHOLD = 1.1


def main():
    p = argparse.ArgumentParser(description="Check WAV with GMM + keyword classifier")
    p.add_argument("wav", help="Path to WAV file")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target GMM model path")
    p.add_argument("--ubm", default=DEFAULT_UBM, help="UBM GMM model path")
    p.add_argument("--kw", default=DEFAULT_KW_MODEL, help="Keyword classifier model (keras) path")
    p.add_argument("--threshold", type=float, default=DEFAULT_SPEAKER_THRESHOLD, help="Optional LLR threshold for GMM decision")
    args = p.parse_args()

    target_gmm = load_model(args.target)
    ubm_gmm = load_model(args.ubm)

    gmm_result = predict_file(args.wav, target_gmm, ubm_gmm, threshold=args.threshold)

    clf = AudioClassifier()
    clf.load_model(args.kw)
    kw_pred, kw_score = clf.predict(args.wav)

    out = {
        "file": args.wav,
        "gmm": {
            "llr": gmm_result.get("llr"),
            "score_target": gmm_result.get("score_target"),
            "score_ubm": gmm_result.get("score_ubm"),
            "pred": gmm_result.get("pred"),
        },
        "keyword_classifier": {
            "pred": kw_pred,
            "score": float(kw_score) if kw_score is not None else None,
        },
    }

    print(json.dumps(out, indent=2))
    if out["gmm"]["pred"] and out["keyword_classifier"]["pred"]:
        print("Access Granted")
    elif out["gmm"]["pred"] and not out["keyword_classifier"]["pred"]:
        print("Access Denied: Correct speaker but keyword not stated")
    elif not out["gmm"]["pred"] and out["keyword_classifier"]["pred"]:
        print("Access Denied: Incorrect speaker with keyword stated")
    else:
        print("Access Denied: Incorrect speaker and keyword not stated")


if __name__ == "__main__":
    main()