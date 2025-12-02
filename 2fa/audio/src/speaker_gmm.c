#include "speaker_gmm.h"

// little bit scuffed but works for now :P
#include "../models/gmm_params.inc"
#include <stdint.h>

#define Q_FEATURE Q_MEANS

typedef const int8_t gmm_row8_t[GMM_DIMENSION];
typedef const int16_t gmm_row16_t[GMM_DIMENSION];
typedef const int32_t gmm_row32_t[GMM_DIMENSION];

typedef struct {
  gmm_row8_t *means;
  gmm_row32_t *inv_covs;
  const int16_t *log_consts;
} GMM;

typedef const double gmm_rowd_t[GMM_DIMENSION];

typedef struct {
  gmm_rowd_t *means;
  gmm_rowd_t *inv_covs;
  const double *log_consts;
} GMMd;

int64_t gmm_log_likelihood(GMM gmm, int16_t *x) {
  int64_t max_term = INT64_MIN;

  for (int k = 0; k < K; k++) {
    int64_t sum_sq = 0;

    for (int d = 0; d < D; d++) {
      int64_t diff = (int64_t)(x[d]) - (int64_t)(gmm.means[k][d]);
      sum_sq += diff * diff * (int64_t)(gmm.inv_covs[k][d]);
      // diff is Q6, inv_covs is Q11, so output is Q23
    }

    sum_sq >>= 16; // convert to Q7
    sum_sq /= 2;   // divide by 2, still Q7
    int64_t term = (int64_t)gmm.log_consts[k] - sum_sq;
    if (term > max_term)
      max_term = term; // approximate log-sum-exp
  }
  return max_term; // approximate log-likelihood
}

double gmmd_log_likelihood(GMMd gmm, double *x) {
  double max_term = -1e30;

  for (int k = 0; k < K; k++) {
    double sum_sq = 0.0;

    for (int d = 0; d < D; d++) {
      double diff = x[d] - gmm.means[k][d];
      sum_sq += diff * diff * gmm.inv_covs[k][d];
    }

    sum_sq /= 2.0;
    double term = gmm.log_consts[k] - sum_sq;
    if (term > max_term)
      max_term = term; // approximate log-sum-exp
  }
  return max_term; // approximate log-likelihood
}

int64_t ubm_gmm_log_likelihood(int16_t *feature_vector) {
  static GMM ubm_gmm = {
      .means = (gmm_row8_t *)ubm_means,
      .inv_covs = (gmm_row32_t *)ubm_inv_covs,
      .log_consts = ubm_log_consts,
  };
  return gmm_log_likelihood(ubm_gmm, feature_vector);
}

double ubm_gmmd_log_likelihood(double *feature_vector) {
  static GMMd ubm_gmmd = {
      .means = (gmm_rowd_t *)ubm_means_d,
      .inv_covs = (gmm_rowd_t *)ubm_inv_covs_d,
      .log_consts = ubm_log_consts_d,
  };
  return gmmd_log_likelihood(ubm_gmmd, feature_vector);
}

int64_t target_gmm_log_likelihood(int16_t *feature_vector) {
  static GMM target_gmm = {
      .means = (gmm_row8_t *)target_means,
      .inv_covs = (gmm_row32_t *)target_inv_covs,
      .log_consts = target_log_consts,
  };
  return gmm_log_likelihood(target_gmm, feature_vector);
}

int64_t target_speaker_llr(int16_t *feature_vector) {
  int64_t target_ll = target_gmm_log_likelihood(feature_vector);
  int64_t ubm_ll = ubm_gmm_log_likelihood(feature_vector);
  return target_ll - ubm_ll;
}

double int64_to_double_loglikelihood(int64_t ll_int) {
  return (double)ll_int / (1 << 7);
}
