#include "speaker_gmm.h"

#include "../models/gmm_params.inc"
#include <stdint.h>

#define Q_FEATURE Q_MEANS

typedef const int8_t gmm_row_t[GMM_DIMENSION];

typedef struct {
  gmm_row_t *means;
  gmm_row_t *inv_covs;
  const int16_t *log_consts;
} GMM;

int64_t gmm_log_likelihood(GMM gmm, int16_t *x) {
  int64_t max_term = INT64_MIN;
  for (int k = 0; k < K; k++) {
    int64_t sum_sq = 0;
    for (int d = 0; d < D; d++) {
      int64_t diff = (int64_t)(x[d]) - (int64_t)(gmm.means[k][d]);
      // pad to avoid losing precision in multiplication
      diff <<= 5;
      sum_sq += (diff * diff * (int64_t)(gmm.inv_covs[k][d]));
      // diff is Q11, ubminv_covs is Q-13, so output is Q9
    }
    sum_sq >>= 1; // divide by 2, still Q9
    sum_sq >>= 2; // convert to Q7
    int64_t term = (int64_t)(gmm.log_consts[k]) - sum_sq;
    if (term > max_term) {
      max_term = term; // approximate log-sum-exp
    }
  }
  return max_term; // approximate log-likelihood
}

int64_t ubm_gmm_log_likelihood(int16_t *feature_vector) {
  static GMM ubm_gmm = {
      .means = (gmm_row_t *)ubm_means,
      .inv_covs = (gmm_row_t *)ubm_inv_covs,
      .log_consts = ubm_log_consts,
  };
  return gmm_log_likelihood(ubm_gmm, feature_vector);
}

int64_t target_gmm_log_likelihood(int16_t *feature_vector) {
  static GMM target_gmm = {
      .means = (gmm_row_t *)target_means,
      .inv_covs = (gmm_row_t *)target_inv_covs,
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
