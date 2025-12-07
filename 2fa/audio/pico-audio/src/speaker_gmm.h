#ifndef SPEAKER_GMM_H
#define SPEAKER_GMM_H

#include <stdbool.h>
#include <stdint.h>

#define GMM_DIMENSION 13
#define GMM_MIXTURES 32

// calculates log likelihood of input frame based on target gmm
// input vector must be of length GMM_DIMENSION = 39
int64_t target_gmm_log_likelihood(int16_t *feature_vector);

// calculates log likelihood of input frame based on ubm gmm
// input vector must be of length GMM_DIMENSION = 39
int64_t ubm_gmm_log_likelihood(int16_t *feature_vector);

// calculates log likelihood ratio of input frame
int64_t target_speaker_llr(int16_t *feature_vector);

// debug for time being
double int64_to_double_loglikelihood(int64_t ll_int);
float int64_to_float_loglikelihood(int64_t ll_int);

#ifdef DOUBLE_GMM

// debug
double ubm_gmmd_log_likelihood(double *feature_vector);

#endif

void float_to_g6int16_arr(float *input, int16_t *output, int length);

int64_t mfcc_target_speaker_llr_mean(float *mfcc_feats, int num_frames);

int classify_speaker(float *mfcc_feats, int num_frames);

#endif
