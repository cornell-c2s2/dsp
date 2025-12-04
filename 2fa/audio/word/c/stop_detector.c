// stop_detector.c
#include "stop_detector.h"

#include "mfcc.h"                        // compute_mfcc
#include "audio_classifier_inference.h"  // audio_classifier_predict
#include "model_params.h"                // for INPUT_SIZE

// Use the same max frames as in Python (max_length = 1000)
#define MAX_FRAMES 500

// This function takes a raw audio buffer and runs MFCC + neural net.
float classify_signal(const float *signal, int num_samples)
{
    // 1) Compute MFCCs in frame-major order:
    //    [c0..c12 for frame0, then frame1, ...]
    float mfcc_frame_major[MAX_FRAMES * MFCC_N_MFCC];

    int num_frames = compute_mfcc(signal,
                                  num_samples,
                                  mfcc_frame_major,
                                  MAX_FRAMES);

    // Clamp number of frames to MAX_FRAMES (same as Python max_length)
    int F      = MFCC_N_MFCC;    // 13
    int max_T  = MAX_FRAMES;     // 1000
    int T      = num_frames;
    if (T > max_T) {
        T = max_T;
    }

    // 2) Reorder into coefficient-major layout to match Python:
    //    Python flatten() on shape (F, max_T):
    //      [c0(t0..max_T-1), c1(t0..max_T-1), ...]
    //    Our frame-major buffer is shape (max_T, F):
    //      frame t: mfcc_frame_major[t*F + c]
    //
    //    We build mfcc_feats of length INPUT_SIZE = F * max_T.
    float mfcc_feats[INPUT_SIZE];

    for (int c = 0; c < F; ++c) {
        for (int t = 0; t < max_T; ++t) {
            float val = 0.0f;
            if (t < T) {
                // frame-major index: frame t, coeff c
                val = mfcc_frame_major[t * F + c];
            }
            // coefficient-major index: coeff c, time t
            mfcc_feats[c * max_T + t] = val;
        }
    }

    // 3) Run through the neural net
    float prob = audio_classifier_predict(mfcc_feats);
    return prob;
}

int classify_signal_binary(const float *signal,
                           int num_samples,
                           float threshold)
{
    float p = classify_signal(signal, num_samples);
    return (p > threshold) ? 1 : 0;
}
