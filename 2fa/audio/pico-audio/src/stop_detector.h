// stop_detector.h
#ifndef STOP_DETECTOR_H
#define STOP_DETECTOR_H

#include <stdint.h>

// signal: mono float samples at 16 kHz
// num_samples: length of the signal buffer
// Returns probability that the clip contains "stop" (0..1)
float classify_signal(const float *signal, int num_samples);

// Convenience: thresholded version, returns 1 if "stop" detected, else 0
int classify_signal_binary(const float *signal,
                           int num_samples,
                           float threshold);

#endif // STOP_DETECTOR_H
