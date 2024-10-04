#include <iir.h>

// Save 20 bytes for coefficients... 2028 bytes left in memory so far
int8_t *b_coefficients[5];
int8_t *a_coefficients[5];

int8_t *gen_IIR(int8_t *pcm, int pcm_size, int8_t *target, int target_size, int8_t *output)
{
  // Another 10 bytes... 2018 bytes left
  int8_t *history[5];

  for (int i = 0; i < pcm_size; i++)
  {
    //  TODO: possible issue of overflow?
    int8_t p_sum = 0;
    for (int j = 0; j < 4; j++)
    {
      p_sum += b_coefficients[j] * pcm[i - j];
    }
    int8_t q_sum = 0;
    for (int j = 1; j < 4; j++)
    {
      q_sum += a_coefficients[j] * output[i - j];
    }
    output[i] = p_sum + q_sum;
  }
}

/*

   Command line: ./mkfilter -Bu -Lp -o 4 -a 0.2 -l

#define NZEROS 4
#define NPOLES 4
#define GAIN   2.146710182e+01

static float xv[NZEROS+1], yv[NPOLES+1];

static void filterloop()
  { for (;;)
      { xv[0] = xv[1]; xv[1] = xv[2]; xv[2] = xv[3]; xv[3] = xv[4];
        xv[4] = `next input value' / GAIN;
        yv[0] = yv[1]; yv[1] = yv[2]; yv[2] = yv[3]; yv[3] = yv[4];
        yv[4] =   (xv[0] + xv[4]) + 4 * (xv[1] + xv[3]) + 6 * xv[2]
                     + ( -0.0301188750 * yv[0]) + (  0.1826756978 * yv[1])
                     + ( -0.6799785269 * yv[2]) + (  0.7820951980 * yv[3]);
        `next output value' = yv[4];
      }
  }


*/