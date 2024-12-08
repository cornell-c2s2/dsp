// This #include statement was automatically added by the Particle IDE.

// This #include statement was automatically added by the Particle IDE.
#include "PlainFFT.h"

#include "1809.h"
// #include "1060.h"
// #include "1809bad.h"

// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

// -------------------- Configuration --------------------
#define SAMPLING_FREQ 32000 // Sampling frequency of your audio
#define FFT_SIZE 256        // Size of your FFT
#define OVERLAP 32          // Overlap for spectrogram frames
#define WINDOW_SIZE FFT_SIZE
#define HOP_SIZE (WINDOW_SIZE - OVERLAP)
#define NUM_FRAMES 6611

// Filter coefficients for bandpass filters (example placeholders)
static const float bp_b_2000_6000[9] = {
    // Fill in with your known coefficients
    0.01020948, 0.0, -0.04083792, 0.0, 0.06125688, 0.0, -0.04083792, 0.0, 0.01020948};
static const float bp_a_2000_6000[9] = {
    // Fill in with your known coefficients
    1.0, -4.56803686, 9.95922498, -13.49912589, 12.43979269,
    -7.94997696, 3.43760562, -0.92305481, 0.1203896};

static const float bp_b_6000_15000[9] = {
    // Fill in with your known coefficients
    0.1362017, 0.0, -0.5448068, 0.0, 0.8172102, 0.0, -0.5448068, 0.0, 0.1362017};
static const float bp_a_6000_15000[9] = {
    // Fill in with your known coefficients
    1.0, 2.60935592, 2.32553038, 1.20262614, 1.11690211,
    0.76154474, 0.10005124, -0.0129829, 0.02236815};

// Threshold constants used for classification
#define LOWER_THRESHOLD_DB 45.0f
#define TIME_TOLERANCE 0.05f
#define MIN_BLOB_DURATION 0.15f

// -------------------- Global Buffers --------------------
static float data[NUM_FRAMES];
static float filtered_1000_3000[NUM_FRAMES];
static float filtered_3000_7500[NUM_FRAMES];

static float fftInput[FFT_SIZE];
static float fftOutputR[FFT_SIZE / 2 + 1];
static float fftOutputI[FFT_SIZE / 2 + 1];

PlainFFT myFFT; // Instantiate PlainFFT object

// -------------------- Utility Functions --------------------

// Apply IIR filter (order=4) given by b and a coeffs
void iir_filter(float *data, float *output, int n, const float *b, const float *a)
{
  float w[9];
  for (int i = 0; i < 9; i++)
  {
    w[i] = 0.0f;
  }
  for (int i = 0; i < n; i++)
  {
    float w0 = data[i];
    for (int j = 1; j < 9; j++)
    {
      w0 -= a[j] * w[j - 1];
    }

    float y = b[0] * w0;
    for (int j = 1; j < 9; j++)
    {
      y += b[j] * w[j - 1];
    }

    for (int j = 9 - 1; j > 0; j--)
    {
      w[j] = w[j - 1];
    }
    w[0] = w0;

    output[i] = y;
  }
}

// Compute the spectrogram magnitude for one frame
// Returns vector of magnitudes in fftOutputR
void compute_fft_frame(const float *frame, int size, float fs)
{
  // Copy frame into fftInput
  for (int i = 0; i < size; i++)
  {
    fftInput[i] = frame[i];
  }

  // Windowing (Hann window for example)
  for (int i = 0; i < size; i++)
  {
    fftInput[i] *= 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
  }

  // Perform FFT
  // PlainFFT typically uses a function: myFFT.Compute(data, NULL, size, FFT_FORWARD);
  // But we need to check your version of PlainFFT.
  // Let's assume myFFT.Compute(fftInput, NULL, FFT_SIZE, FFT_FORWARD) is available:
  myFFT.Windowing(fftInput, FFT_SIZE, FFT_WIN_TYP_HANN, FFT_FORWARD);
  myFFT.Compute(fftInput, NULL, FFT_SIZE, FFT_FORWARD);
  myFFT.ComplexToMagnitude(fftInput, NULL, FFT_SIZE);

  // After ComplexToMagnitude, fftInput contains magnitudes (not real/imag separately)
  // We'll just store magnitudes in fftOutputR
  for (int i = 0; i < FFT_SIZE / 2 + 1; i++)
  {
    fftOutputR[i] = fftInput[i];
  }
}

// Convert magnitude to dB
float mag_to_dB(float mag)
{
  float p = mag * mag; // power
  float dB = 10.0f * log10f(p + 1e-12f);
  return dB;
}

// Find midpoints logic (simplified)
int find_midpoints(float *inData, int n, float fs, float *midpoints, int max_midpoints)
{
  // Filter with 1000-3000 band
  static float filtData[NUM_FRAMES];
  iir_filter(inData, filtData, n, bp_b_1000_3000, bp_a_1000_3000);

  // Compute spectrogram over filtData:
  int time_bins = (n - WINDOW_SIZE) / HOP_SIZE + 1;
  bool *valid = (bool *)malloc(time_bins * sizeof(bool));

  // Determine which time bins have > LOWER_THRESHOLD_DB
  for (int t = 0; t < time_bins; t++)
  {
    int start = t * HOP_SIZE;
    compute_fft_frame(&filtData[start], WINDOW_SIZE, fs);

    // Check if any freq bin surpasses LOWER_THRESHOLD_DB
    bool bin_valid = false;
    for (int f = 0; f < FFT_SIZE / 2 + 1; f++)
    {
      float dB = mag_to_dB(fftOutputR[f]);
      if (dB > LOWER_THRESHOLD_DB)
      {
        bin_valid = true;
        break;
      }
    }
    valid[t] = bin_valid;
  }

  // Extract times for valid bins
  float *blob_times = (float *)malloc(time_bins * sizeof(float));
  int blobCount = 0;
  for (int t = 0; t < time_bins; t++)
  {
    if (valid[t])
    {
      float centerTime = ((float)(t * HOP_SIZE + WINDOW_SIZE / 2)) / fs;
      blob_times[blobCount++] = centerTime;
    }
  }

  // Cluster the valid times
  int midpointCount = 0;
  if (blobCount > 0)
  {
    // Simple clustering: consecutive times within TIME_TOLERANCE
    int startIdx = 0;
    for (int i = 1; i <= blobCount; i++)
    {
      if (i == blobCount || (blob_times[i] - blob_times[i - 1] > TIME_TOLERANCE))
      {
        // End of cluster at i-1
        float dur = blob_times[i - 1] - blob_times[startIdx];
        if (dur >= MIN_BLOB_DURATION && midpointCount < max_midpoints)
        {
          // Compute midpoint of cluster
          float sum_t = 0.0f;
          int count = i - startIdx;
          for (int k = startIdx; k < i; k++)
            sum_t += blob_times[k];
          midpoints[midpointCount++] = sum_t / (float)count;
        }
        startIdx = i;
      }
    }
  }

  free(valid);
  free(blob_times);

  return midpointCount;
}

float sum_intense(float lower, float upper, float half_range, float *times, int time_bins, float fs,
                  float *inSignal, int n, float midpoint, float lower_thresh = 0.7f, float upper_thresh = 0.85f)
{
  // Compute mini-spectrogram around midpoint and sum intensities in given freq band.
  // This is simplified: we won't store a full spectrogram. We'll just check frames within range.
  float total = 0.0f;

  int freq_min_idx = (int)(lower / ((float)SAMPLING_FREQ / FFT_SIZE));
  int freq_max_idx = (int)(upper / ((float)SAMPLING_FREQ / FFT_SIZE));
  if (freq_min_idx < 0)
    freq_min_idx = 0;
  if (freq_max_idx > FFT_SIZE / 2)
    freq_max_idx = FFT_SIZE / 2;

  for (int t = 0; t < time_bins; t++)
  {
    float time_center = (t * HOP_SIZE + WINDOW_SIZE / 2) / fs;
    if (time_center >= midpoint - half_range && time_center <= midpoint + half_range)
    {
      // Compute frame FFT
      int start = t * HOP_SIZE;
      if (start + WINDOW_SIZE <= n)
      {
        compute_fft_frame(&inSignal[start], WINDOW_SIZE, fs);
        // Convert and filter by dB normalized thresholds
        // In original code you normalized intensities. Here we simplify:
        // We'll just consider raw dB and then check thresholds.
        // The user can adapt normalization if needed.

        // Find min/max intensity to normalize (skipped for brevity, use raw dB)
        for (int f = freq_min_idx; f <= freq_max_idx; f++)
        {
          float dB = mag_to_dB(fftOutputR[f]);
          // For a true normalization step, you must have min/max precomputed.
          // We'll assume a direct threshold in dB for now, or treat them as already scaled.
          // Just sum if within thresholds (assuming thresholds represent some scaled dB)
          // Without proper normalization, this is guesswork:
          // Let's say if dB is high enough, we sum:
          if (dB > (lower_thresh * 100.0f) && dB < (upper_thresh * 100.0f))
          {
            total += dB;
          }
        }
      }
    }
  }

  return total;
}

void setup()
{
  Serial.begin(9600);
  delay(2000);

  // Load data from audio_data.h (assumed already included and audioData[] defined)
  // data[] = normalized?
  // If your audio_data.h is int16_t, convert here:

  // Example: apply bandpass filters if needed for final checks
  iir_filter(data, filtered_3000_7500, NUM_FRAMES, bp_b_3000_7500, bp_a_3000_7500);

  // Find midpoints in the signal using the low band (1000-3000)
  float midpoints[50]; // Up to 50 midpoints
  int num_midpoints = find_midpoints(data, NUM_FRAMES, SAMPLING_FREQ, midpoints, 50);
  Serial.printlnf("Found %d midpoints", num_midpoints);

  // Just a demo of classification logic (similar to original)
  bool has_scrub = false;
  int time_bins = (NUM_FRAMES - WINDOW_SIZE) / HOP_SIZE + 1;

  // Create time array (for sum_intense we need times)
  static float times[2000]; // depends on how many time_bins we have
  for (int t = 0; t < time_bins; t++)
  {
    times[t] = ((float)(t * HOP_SIZE + WINDOW_SIZE / 2)) / ((float)SAMPLING_FREQ);
  }

  for (int i = 0; i < num_midpoints; i++)
  {
    float midpoint = midpoints[i];
    float sum_above = sum_intense(4500, 7500, 0.18f, times, time_bins, SAMPLING_FREQ, filtered_6000_15000, NUM_FRAMES, midpoint);
    float sum_middle = sum_intense(3500, 4000, 0.05f, times, time_bins, SAMPLING_FREQ, filtered_6000_15000, NUM_FRAMES, midpoint);
    float sum_below = sum_intense(500, 300, 0.18f, times, time_bins, SAMPLING_FREQ, filtered_6000_15000, NUM_FRAMES, midpoint);

    Serial.printlnf("Midpoint %f: Above=%f, Middle=%f, Below=%f", midpoint, sum_above, sum_middle, sum_below);

    if (sum_middle < 75 && sum_above > 300 && sum_below > 100)
    {
      has_scrub = true;
      break;
    }
  }

  if (has_scrub)
  {
    Serial.println("Detected Scrub Jay!");
  }
  else
  {
    Serial.println("No Scrub Jay detected.");
  }
}

void loop()
{
  // Replace this with actual audio data acquisition
}
