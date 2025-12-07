// stop_detector.h
#ifndef STOP_DETECTOR_H
#define STOP_DETECTOR_H

#include <stdint.h>

typedef struct {
  float stop_prob;
  float speaker_llr;
} classifier_output_t;
;

typedef struct {
  int stop;
  int speaker;
} classifier_binary_output_t;

// signal: mono float samples at 16 kHz
// num_samples: length of the signal buffer
// Returns probability that the clip contains "stop" (0..1)
classifier_output_t classify_signal(const float *signal, int num_samples);

// Convenience: thresholded version, returns 1 if "stop" detected, else 0
classifier_binary_output_t classify_signal_binary(const float *signal,
                                                  int num_samples,
                                                  float stop_threshold,
                                                  float speaker_threshold);

#endif // STOP_DETECTOR_H
