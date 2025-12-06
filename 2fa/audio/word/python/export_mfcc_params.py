import numpy as np
import librosa

# Must match what you used in AudioClassifier.extract_mfcc()
SAMPLE_RATE = 16000
N_FFT = 512
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 13

OUTPUT_HEADER = "../c/mfcc_params.h"


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


def dct_ortho(n_mfcc, n_mels):
    """
    Build a DCT-II, orthonormal basis matrix of shape (n_mfcc, n_mels),
    matching scipy.fftpack.dct(..., type=2, norm='ortho') used by librosa.
    """
    basis = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    # k = 0 row
    basis[0, :] = np.sqrt(1.0 / n_mels)
    # k >= 1 rows
    n = np.arange(n_mels, dtype=np.float32)
    for k in range(1, n_mfcc):
        basis[k, :] = np.sqrt(2.0 / n_mels) * np.cos(
            np.pi * (n + 0.5) * k / n_mels
        )
    return basis


def main():
    # Hann window (same as window='hann', fftbins=True)
    hann = librosa.filters.get_window("hann", FRAME_LENGTH, fftbins=True)

    # Mel filterbank: shape (n_mels, n_fft//2 + 1)
    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2.0,
        htk=True,
        norm=None,
    )

    # DCT matrix: shape (n_mfcc, n_mels)
    dct_mat = dct_ortho(N_MFCC, N_MELS)

    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef MFCC_PARAMS_H\n")
        f.write("#define MFCC_PARAMS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"#define MFCC_SAMPLE_RATE {SAMPLE_RATE}\n")
        f.write(f"#define MFCC_N_FFT {N_FFT}\n")
        f.write(f"#define MFCC_FRAME_LENGTH {FRAME_LENGTH}\n")
        f.write(f"#define MFCC_HOP_LENGTH {HOP_LENGTH}\n")
        f.write(f"#define MFCC_N_MELS {N_MELS}\n")
        f.write(f"#define MFCC_N_MFCC {N_MFCC}\n")
        f.write(f"#define MFCC_N_FREQ_BINS {(N_FFT//2) + 1}\n\n")

        write_array(f, "float", "HANN_WINDOW", hann)
        write_array(f, "float", "MEL_FILTER", mel_fb)
        write_array(f, "float", "DCT_MATRIX", dct_mat)

        f.write("#endif // MFCC_PARAMS_H\n")

    print(f"Wrote {OUTPUT_HEADER}")


if __name__ == "__main__":
    main()
