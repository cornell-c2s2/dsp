#include "speaker_gmm.h"
#include <stdint.h>
#include <stdio.h>

int main() {
  int16_t x[GMM_DIMENSION];
  for (int d = 0; d < GMM_DIMENSION; d++)
    x[d] = 0;

  int64_t ll = ubm_gmm_log_likelihood(x);
  printf("Approximate log-likelihood: %ld\n", ll);
  printf("Approximate log-likelihood (float): %f\n", (double)ll / (1 << 7));
  return 0;
}
