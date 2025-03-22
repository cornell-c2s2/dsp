// This #include statement was automatically added by the Particle IDE.
#include "PlainFFT.h"

// This #include statement was automatically added by the Particle IDE.
#include "classifier.h"


/*
 * Project donut-microphone
 * Author: C2S2 Software
 * Date:
 * For comprehensive documentation and examples, please visit:
 * https://docs.particle.io/firmware/best-practices/firmware-template/
 */

// Include Particle Device OS APIs
#include "Particle.h"
#include "classifier.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

static unsigned long lastSampleTime = 0;  // Time of last sample
const int sampleRate = 16000;             // Actually closer to 9000 (hardware limitations?)
int count = 0;                            // Number of samples in buffer
const int BUF_SIZE = 3807;                // Size of the buffer
static float buffer[BUF_SIZE];            // Collect samples for the classifier
const int UPBUF_SIZE = BUF_SIZE * 16 / 9; // 9000 Hz to 16000 Hz
static float upsampledBuffer[UPBUF_SIZE]; // Buffer after upsampling
bool flag = false;                        // Hit noise requirement

// Alert user we are waiting for noise
void countdown(bool countdown)
{
  if (!countdown)
  {
    Serial.println("3");
    delay(1000);
    Serial.println("2");
    delay(1000);
    Serial.println("1");
    delay(1000);
  }
  Serial.println("LISTENING...");
}

void setup()
{
  Serial.begin(115200);
  Serial1.begin(115200);
  delay(5000);
  countdown(false);
}
// Upsample using linear interpolation
void upsampleLinear(const float *inBuffer, int oldSize, float *outBuffer, int newSize)
{
  for (int i = 0; i < newSize; i++)
  {
    float oldIndex = i * ((float)(oldSize - 1) / (float)(newSize - 1));

    int indexFloor = (int)floor(oldIndex);
    int indexCeil = (indexFloor == (oldSize - 1)) ? (oldSize - 1) : (indexFloor + 1);

    float frac = oldIndex - indexFloor;

    float valFloor = inBuffer[indexFloor];
    float valCeil = inBuffer[indexCeil];
    outBuffer[i] = valFloor + (valCeil - valFloor) * frac;
  }
}

void loop()
{
  // Check if it has been enough time since last sample
  unsigned long currentTime = micros();
  if (currentTime - lastSampleTime >= (1000000 / sampleRate))
  {
    // Serial.printf("%d", currentTime - lastSampleTime);
    // Serial.print(",");
    lastSampleTime = currentTime;
  Serial1.print('0');
    // Read 12-bit ADC value from A0
    int adcValue = analogRead(A0);
    // Check if the digital signal is sufficient far from 2048 (we have noise) and let the sample collection start
    if (adcValue < 1548 || adcValue > 2548)
    {
      // Serial.println(adcValue);
      flag = true;
    }
    if (flag)
    {
      // Convert 12-bit ADC value to signed 16-bit PCM
      int16_t pcmSample = (adcValue - 2048) * 16;

      if (count < BUF_SIZE) // Collect data
      {
        buffer[count++] = pcmSample;
      }
      else if (count == BUF_SIZE) // Data collected
      {
        upsampleLinear(buffer, BUF_SIZE, upsampledBuffer, UPBUF_SIZE);

        // Normalize the data to [-1, 1]
        for (int i = 0; i < UPBUF_SIZE; i++)
        {
          upsampledBuffer[i] = upsampledBuffer[i] / 32768.0;
        }

        Serial.println("Begin Classification...");
        classify(upsampledBuffer, (sizeof(upsampledBuffer) / sizeof(upsampledBuffer[0])));
        // Serial.println(micros() - before);
        Serial.println("Classification Ended!");
        Serial.println("");

        // Reset after classification
        flag = false;
        count = 0;
        countdown(true);
      }
    }
  }
}