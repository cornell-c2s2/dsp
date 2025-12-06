#include "speaker_gmm.h"
#include <stdint.h>
#include <stdio.h>

int main() {
#ifdef DOUBLE_GMM
  double x[GMM_DIMENSION];
  for (int d = 0; d < GMM_DIMENSION; d++) {
    x[d] = 0.125 * (d + 1);
  }

  double ll = ubm_gmmd_log_likelihood(x);
  printf("Approximate log-likelihood: %f\n", ll);
#endif

  int16_t x2[GMM_DIMENSION];
  for (int16_t d = 0; d < GMM_DIMENSION; d++) {
    x2[d] = (int16_t)((d + 1) * 8);
  }

  int64_t ll_int = ubm_gmm_log_likelihood(x2);
  printf("Approximate log-likelihood (int): %ld\n", ll_int);
  printf("Converted to double: %f\n", int64_to_double_loglikelihood(ll_int));
  int64_t llr = target_speaker_llr(x2);
  printf("Approximate LLR (int): %ld\n", target_speaker_llr(x2));
  printf("As double: %f\n", int64_to_double_loglikelihood(llr));
  return 0;
}
