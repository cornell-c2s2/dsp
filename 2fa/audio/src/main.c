#include "speaker_gmm.h"
#include <stdint.h>
#include <stdio.h>

int main() {
  double x[GMM_DIMENSION];
  for (int d = 0; d < GMM_DIMENSION; d++)
    x[d] = 0.5;

  int16_t x2[GMM_DIMENSION];
  for (int d = 0; d < GMM_DIMENSION; d++)
    x2[d] = 32;

  double ll = ubm_gmmd_log_likelihood(x);
  printf("Approximate log-likelihood: %f\n", ll);
  int64_t ll_int = ubm_gmm_log_likelihood(x2);
  printf("Approximate log-likelihood (int): %ld\n", ll_int);
  printf("Converted to double: %f\n", int64_to_double_loglikelihood(ll_int));
  return 0;
}
