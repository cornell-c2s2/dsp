#include <stdint.h>
/// @brief Implementation of a general IIR.
/// @param pcm PCM samples of the bird call
/// @param pcm_size size of the pcm array
/// @param target template signal for the signal
/// @param target_size size of the target array
/// @param output pointer holding the contents of the result
int8_t *gen_IIR(int8_t *pcm, int pcm_size, int8_t *target, int target_size, int8_t *output);
