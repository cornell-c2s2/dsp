import numpy as np
import librosa

from keyword_classifier import AudioClassifier

AUDIO_PATH = "../../data/testing/stop_121417.wav" 
#AUDIO_PATH = "../../data/testing/bed__common_voice_en_82827.wav" 
OUTPUT_HEADER = "../c/test_mfcc.h"

def write_array(f, c_type, name, arr, per_line=8):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    f.write(f"static const {c_type} {name}[{arr.size}] = {{\n    ")
    for i, v in enumerate(arr):
        f.write(f"{float(v):.8e}f")
        if i != arr.size - 1:
            f.write(", ")
        if (i + 1) % per_line == 0:
            f.write("\n    ")
    f.write("\n};\n\n")

def main():
    clf = AudioClassifier(n_mfcc=13, max_length=500)
    mfcc_flat = clf.extract_mfcc(AUDIO_PATH)
    if mfcc_flat is None:
        raise RuntimeError("Failed to extract MFCCs")

    print("MFCC length:", mfcc_flat.shape[0])  # should be INPUT_SIZE

    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef TEST_MFCC_H\n")
        f.write("#define TEST_MFCC_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define TEST_MFCC_SIZE {mfcc_flat.size}\n\n")

        write_array(f, "float", "TEST_MFCC", mfcc_flat)

        f.write("#endif // TEST_MFCC_H\n")

    print(f"Wrote {OUTPUT_HEADER}")

if __name__ == "__main__":
    main()
