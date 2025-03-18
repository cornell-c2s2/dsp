
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
const int sampleRate = 16000; // 16 KHz
int count = 0;
const int BUF_SIZE = 10000;
float buffer[BUF_SIZE];

void setup()
{
  Serial.begin(115200);
  delay(5000);
  Serial.println("3");
  delay(1000);
  Serial.println("2");
  delay(1000);
  Serial.println("1");
  delay(1000);
  Serial.println("SPEAK!!!");
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
      buffer[count++] = pcmSample / 32768.0;
      // buffer[count++] = adcValue;
    }
    else if (count == BUF_SIZE)
    {
      count++;
      classify(buffer);
      Serial.print("PASSED");
      for (int i = 0; i < BUF_SIZE; i++)
      {
        Serial.printf("%f,", buffer[i]);
      }
    }
    // Serial.print(pcmSample / 32768.0);
    // Serial.print(",");
  }
}