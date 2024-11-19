#include <fftw3.h>
#include <stdio.h>

int main()
{
  int nfft = 1024;
  fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * nfft);
  if (in == NULL)
  {
    printf("FFTW malloc failed\n");
    return -1;
  }
  printf("FFTW malloc succeeded\n");
  fftw_free(in);
  return 0;
}
