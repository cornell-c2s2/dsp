#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <stdbool.h>

// Function prototypes
bool butter_bandpass(float lowcut, float highcut, float *b, float *a);
void butter_bandpass_filter(float *data, int n, float *b, float *a, float *output);
void compute_spectrogram(float *signal, int signal_length, int fs, float **frequencies, float **times, float ***Sxx, int *freq_bins, int *time_bins);
float normalize_intensity(float value, float min, float max);
float sum_intense(float lower, float upper, float half_range, float *frequencies, int freq_bins, float *times, int time_bins, float **intensity_dB_filtered, float midpoint);
float *find_midpoints(float *data, int num_frames, int samplingFreq, int *num_midpoints);
void classify(float *data, int data_size);

#endif // CLASSIFIER_H
