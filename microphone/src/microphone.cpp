
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

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

static unsigned long lastSampleTime = 0;
const int sampleRate = 16000;
int count = 0;
const int BUF_SIZE = 3807;
static float buffer[BUF_SIZE];
const int UPBUF_SIZE = BUF_SIZE * 16 / 9;
static float upsampledBuffer[UPBUF_SIZE];
bool flag = false;

void countdown(bool s)
{
  if (!s)
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
  delay(5000);
  countdown(false);
}

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

  unsigned long currentTime = micros();
  if (currentTime - lastSampleTime >= (1000000 / sampleRate))
  {

    // Serial.printf("%d", currentTime - lastSampleTime);
    // Serial.print(",");
    lastSampleTime = currentTime;

    // Read 12-bit ADC value from A0
    int adcValue = analogRead(A0);
    if (adcValue < 1548 || adcValue > 2548)
    {
      // Serial.println(adcValue);
      flag = true;
    }
    if (flag)
    {
      // Convert 12-bit ADC value to signed 16-bit PCM
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
        // unsigned long before = micros();
        classify(upsampledBuffer, (sizeof(upsampledBuffer) / sizeof(upsampledBuffer[0])));
        // Serial.println(micros() - before);
        Serial.println("Classification Ended!");
        Serial.println("");
        flag = false;
        count = 0;
        countdown(true);
      }
    }
  }
}
