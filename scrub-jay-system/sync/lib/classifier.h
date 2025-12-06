#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "PlainFFT.h"
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <cstdio>
#include <cstring>

#define PI 3.14159265358979323846

// Function prototypes
bool butter_bandpass(float lowcut, float highcut, float *b, float *a);
void butter_bandpass_filter(float *data, int n, float *b, float *a, float *output);
void compute_spectrogram(float *signal, int signal_length, int fs, float **frequencies, float **times, float ***Sxx, int *freq_bins, int *time_bins);
float sum_intense(float lower, float upper, float half_range, float *frequencies, int freq_bins, float *times, int time_bins, float **intensity_dB_filtered, float midpoint);
float* find_midpoints(float *data, int num_frames, int samplingFreq, int *num_midpoints);
int classify(float *data, int data_size);

#endif // CLASSIFIER_H