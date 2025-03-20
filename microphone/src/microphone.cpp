
/*
 * Project myProject
 * Author: Your Name
 * Date:
 * For comprehensive documentation and examples, please visit:
 * https://docs.particle.io/firmware/best-practices/firmware-template/
 */

// Include Particle Device OS APIs
#include "Particle.h"
#include "classifier.h"
#include "test.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

// void setup()
// {
//   Serial.begin(115200);
//   delay(5000);
//   classify(data, (sizeof(data) / sizeof(data[0])));
//   Serial.println("PASSED");
// }

// void loop()
// {
// }

// CODE BELOW HERE
static unsigned long lastSampleTime = 0;
const int sampleRate = 16000; // 16 KHz
int count = 0;
const int MOVE_SIZE = 450;
const int BUF_SIZE = 4500;
static float buffer[BUF_SIZE];
const int UPBUF_SIZE = BUF_SIZE * 8 / 5;
static float upsampledBuffer[UPBUF_SIZE];

void countdown()
{
  Serial.println("3");
  delay(1000);
  Serial.println("2");
  delay(1000);
  Serial.println("1");
  delay(1000);
  Serial.println("SQUAWK!!!");
}

void setup()
{
  Serial.begin(115200);
  delay(5000);
  countdown();
}

void upsampleLinear(const float *inBuffer, int oldSize, float *outBuffer, int newSize)
{
  // We assume oldSize > 1
  // We'll treat the old buffer indices as ranging 0..(oldSize-1).
  // We'll map new samples to an "oldIndex" and interpolate.
  for (int i = 0; i < newSize; i++)
  {
    // A float index in the old buffer
    float oldIndex = i * ((float)(oldSize - 1) / (float)(newSize - 1));

    // Floor of oldIndex
    int indexFloor = (int)floor(oldIndex);
    // Cap the ceiling at the last valid index
    int indexCeil = (indexFloor == (oldSize - 1)) ? (oldSize - 1) : (indexFloor + 1);

    // Fractional part for interpolation
    float frac = oldIndex - indexFloor;

    // Linear interpolation
    float valFloor = inBuffer[indexFloor];
    float valCeil = inBuffer[indexCeil];
    outBuffer[i] = valFloor + (valCeil - valFloor) * frac;
  }
}

void loop()
{

  unsigned long currentTime = micros();

  if (currentTime - lastSampleTime >= (1000000 / sampleRate))
  { // ~16 kHz sampling rate

    // Serial.print(currentTime - lastSampleTime);
    // Serial.print(",");
    lastSampleTime = currentTime;

    // Read 12-bit ADC value from A0
    int adcValue = analogRead(A0);

    // Convert 12-bit ADC value (0-4095) to signed 16-bit PCM (-32768 to 32767)
    int16_t pcmSample = (adcValue - 2048) * 16;
    if (count < BUF_SIZE)
    {
      buffer[count++] = pcmSample;
    }
    else if (count == BUF_SIZE)
    {
      upsampleLinear(buffer, BUF_SIZE, upsampledBuffer, UPBUF_SIZE);
      for (int i = 0; i < UPBUF_SIZE; i++)
      {
        upsampledBuffer[i] = upsampledBuffer[i] / 32768.0;
      }
      Serial.println("Begin Classification...");
      classify(upsampledBuffer, (sizeof(upsampledBuffer) / sizeof(upsampledBuffer[0])));
      Serial.println("Classification Ended!");
      Serial.println("");
      count = BUF_SIZE - MOVE_SIZE;
      for (int i = 0; i < BUF_SIZE - MOVE_SIZE; i++)
      {
        buffer[i] = buffer[i + MOVE_SIZE];
      }

      // for (int i = 0; i < UPBUF_SIZE; i++)
      // {
      //   Serial.printf("%f,", upsampledBuffer[i]);
      // }
    }
  }
}
