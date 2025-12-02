#ifndef SPEAKER_GMM_H
#define SPEAKER_GMM_H

#include <stdbool.h>
#include <stdint.h>

#define GMM_DIMENSION 39
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
// hardcoded for now
double int64_to_double_loglikelihood(int64_t ll_int);

// debug
double ubm_gmmd_log_likelihood(double *feature_vector);

#endif
