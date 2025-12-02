#include <stdio.h>
#include "audio_classifier_inference.h"
#include "test_mfcc.h"
#include "model_params.h"

int main(void)
{
    if (TEST_MFCC_SIZE != INPUT_SIZE) {
        printf("Error: TEST_MFCC_SIZE (%d) != INPUT_SIZE (%d)\n",
               TEST_MFCC_SIZE, INPUT_SIZE);
        return 1;
    }

    float prob = audio_classifier_predict(TEST_MFCC);
    printf("C probability: %.8f\n", prob);

    return 0;
}
