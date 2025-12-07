// stop_detector.c
#include "stop_detector.h"

#include "audio_classifier_inference.h" // audio_classifier_predict
#include "mfcc.h"                       // compute_mfcc
#include "model_params.h"               // for INPUT_SIZE
#include "speaker_gmm.h"

// Use the same max frames as in Python (max_length = 500)
#define MAX_FRAMES 500

// This function takes a raw audio buffer and runs MFCC + neural net.
classifier_output_t classify_signal(const float *signal, int num_samples) {
  // 1) Compute MFCCs in frame-major order:
  //    [c0..c12 for frame0, then frame1, ...]
  float mfcc_frame_major[MAX_FRAMES * MFCC_N_MFCC];

  int num_frames =
      compute_mfcc(signal, num_samples, mfcc_frame_major, MAX_FRAMES);

  // Clamp number of frames to MAX_FRAMES (same as Python max_length)
  int F = MFCC_N_MFCC;    // 13
  int max_T = MAX_FRAMES; // 1000
  int T = num_frames;
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

  classifier_output_t out;
  // 3) Run through neural net
  out.stop_prob = audio_classifier_predict(mfcc_feats);
  // 4) Run through speaker GMM
  int64_t speaker_int_prob = mfcc_target_speaker_llr_mean(mfcc_feats, T);
  out.speaker_llr = int64_to_float_loglikelihood(speaker_int_prob);
  return out;
}

classifier_binary_output_t classify_signal_binary(const float *signal,
                                                  int num_samples,
                                                  float stop_threshold,
                                                  float speaker_threshold) {
  classifier_output_t p = classify_signal(signal, num_samples);
  classifier_binary_output_t out;

  out.stop = (p.stop_prob > stop_threshold) ? 1 : 0;

  int64_t speaker_threshold_int = (int64_t)(speaker_threshold * (1 << 8));
  out.speaker = (p.speaker_llr > speaker_threshold_int) ? 1 : 0;
  return out;
}
