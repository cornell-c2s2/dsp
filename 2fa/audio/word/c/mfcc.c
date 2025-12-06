#include "mfcc.h"
#include "mfcc_params.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif


// You must implement or hook this to your FFT library.
// in_time: FRAME_LENGTH real samples
// out_freq: MFCC_N_FFT complex spectrum as interleaved [re0, im0, re1, im1, ...]
void fft_real_forward(const float *in_time, float *out_freq)
{
    const int N = MFCC_N_FFT;

    // Internal working buffers
    static float real[MFCC_N_FFT];
    static float imag[MFCC_N_FFT];

    // Copy input and zero-pad
    for (int i = 0; i < MFCC_FRAME_LENGTH; ++i) {
        real[i] = in_time[i];
        imag[i] = 0.0f;
    }
    for (int i = MFCC_FRAME_LENGTH; i < N; ++i) {
        real[i] = 0.0f;
        imag[i] = 0.0f;
    }

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < N; ++i) {
        if (i < j) {
            float tr = real[i];
            float ti = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tr;
            imag[j] = ti;
        }
        int m = N >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Iterative Cooley–Tukey FFT (DIT)
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * M_PI_F / (float)len;
        float wlen_cos = cosf(ang);
        float wlen_sin = sinf(ang);

        for (int i = 0; i < N; i += len) {
            float w_cos = 1.0f;
            float w_sin = 0.0f;

            for (int k = 0; k < len / 2; ++k) {
                int u = i + k;
                int v = i + k + len / 2;

                float ru = real[u];
                float iu = imag[u];
                float rv = real[v];
                float iv = imag[v];

                // t = w * (rv + i*iv)
                float t_real = w_cos * rv - w_sin * iv;
                float t_imag = w_cos * iv + w_sin * rv;

                // butterfly
                real[v] = ru - t_real;
                imag[v] = iu - t_imag;
                real[u] = ru + t_real;
                imag[u] = iu + t_imag;

                // w *= wlen
                float tmp = w_cos * wlen_cos - w_sin * wlen_sin;
                w_sin     = w_cos * wlen_sin + w_sin * wlen_cos;
                w_cos     = tmp;
            }
        }
    }

    // Export as interleaved complex: [re0, im0, re1, im1, ...]
    for (int k = 0; k < N; ++k) {
        out_freq[2 * k]     = real[k];
        out_freq[2 * k + 1] = imag[k];
    }
}

// Helper to access MEL_FILTER as [mel, freq_bin]
static inline float mel_filter(int mel_idx, int freq_bin) {
    return MEL_FILTER[mel_idx * MFCC_N_FREQ_BINS + freq_bin];
}

// Helper to access DCT_MATRIX as [mfcc_idx, mel_idx]
static inline float dct_coeff(int mfcc_idx, int mel_idx) {
    return DCT_MATRIX[mfcc_idx * MFCC_N_MELS + mel_idx];
}

// Main MFCC computation
int compute_mfcc(const float *signal,
                 int num_samples,
                 float *out_mfcc,
                 int max_frames)
{
    const int frame_len = MFCC_FRAME_LENGTH;
    const int hop_len = MFCC_HOP_LENGTH;
    const int n_fft = MFCC_N_FFT;

    if (num_samples < frame_len || max_frames <= 0) {
        return 0;
    }

    // Buffers for per-frame processing
    float frame[MFCC_FRAME_LENGTH];
    float spectrum[MFCC_N_FFT * 2];  // complex: re, im
    float power_spectrum[MFCC_N_FREQ_BINS];
    float mel_energies[MFCC_N_MELS];
    float log_mel[MFCC_N_MELS];
    float mfcc[MFCC_N_MFCC];

    int frame_idx = 0;
    int out_offset = 0;

    for (;;) {
        int start = frame_idx * hop_len;
        if (start + frame_len > num_samples) {
            break;  // not enough samples for another full frame
        }
        if (frame_idx >= max_frames) {
            break;
        }

        // 1) Frame + Hann window
        for (int i = 0; i < frame_len; ++i) {
            frame[i] = signal[start + i] * HANN_WINDOW[i];
        }

        // 2) FFT (real) -> complex spectrum (n_fft bins)
        // You need to zero-pad frame to n_fft inside your FFT
        fft_real_forward(frame, spectrum);

        // 3) Power spectrum: only first N_FREQ_BINS bins
        for (int k = 0; k < MFCC_N_FREQ_BINS; ++k) {
            float re = spectrum[2 * k];
            float im = spectrum[2 * k + 1];
            power_spectrum[k] = re * re + im * im;
        }

        // 4) Mel filterbank energies
        for (int m = 0; m < MFCC_N_MELS; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < MFCC_N_FREQ_BINS; ++k) {
                sum += mel_filter(m, k) * power_spectrum[k];
            }
            mel_energies[m] = sum;
        }

        // 5) Log-mel energies in dB, mimicking librosa.power_to_db
        //    with ref=np.max, amin=1e-10, top_db=80.

        const float amin   = 1e-10f;
        const float top_db = 80.0f;

        // 5a) Find reference = max mel energy
        float ref = 0.0f;
        for (int m = 0; m < MFCC_N_MELS; ++m) {
            if (mel_energies[m] > ref) {
                ref = mel_energies[m];
            }
        }
        if (ref < amin) {
            ref = amin;
        }
        float log_ref = 10.0f * log10f(ref);

        // 5b) Compute 10 * log10(e) - 10 * log10(ref)
        for (int m = 0; m < MFCC_N_MELS; ++m) {
            float e = mel_energies[m];
            if (e < amin) {
                e = amin;
            }
            float log_e = 10.0f * log10f(e);
            log_mel[m] = log_e - log_ref;
        }

        // 5c) Apply top_db clipping: keep values within [max - 80, max]
        float max_db = log_mel[0];
        for (int m = 1; m < MFCC_N_MELS; ++m) {
            if (log_mel[m] > max_db) {
                max_db = log_mel[m];
            }
        }
        float min_db = max_db - top_db;
        for (int m = 0; m < MFCC_N_MELS; ++m) {
            if (log_mel[m] < min_db) {
                log_mel[m] = min_db;
            }
        }


        // 6) DCT -> MFCC
        for (int c = 0; c < MFCC_N_MFCC; ++c) {
            float sum = 0.0f;
            for (int m = 0; m < MFCC_N_MELS; ++m) {
                sum += dct_coeff(c, m) * log_mel[m];
            }
            mfcc[c] = sum;
        }

        // 7) Store MFCC coefficients in out_mfcc (flattened)
        memcpy(&out_mfcc[out_offset], mfcc, MFCC_N_MFCC * sizeof(float));
        out_offset += MFCC_N_MFCC;
        frame_idx++;
        // if (frame_idx == 1) {
        // printf("First 10 MFCC coeffs of first frame:\n");
        // for (int i = 0; i < 10 && i < MFCC_N_MFCC; ++i) {
        //     printf("%f ", out_mfcc[i]);
        // }
        // printf("\n");
        // }
    }

    return frame_idx; // number of frames computed
}
