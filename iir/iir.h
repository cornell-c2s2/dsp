#include <stdint.h>
/// @brief Implementation of a bandpass butterworth IIR filter using 8 bit integers.
/// Designed to work on the RISCV board.
/// @param pcm PCM samples of the bird call
/// @param pcm_size size of the pcm array
/// @param target template signal for the signal
/// @param target_size size of the target array
/// @param output pointer holding the contents of the result
int8_t *butterworth(int8_t *pcm, int pcm_size, int8_t *target, int target_size, int8_t *output);
