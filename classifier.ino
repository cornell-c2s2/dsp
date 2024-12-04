#include <Arduino.h>
#include <PlainFFT.h>

PlainFFT FFT;

#define NUM_SAMPLES 1024 // Adjust based on available memory
#define WINDOW_SIZE 256
#define NFFT WINDOW_SIZE
#define HOP_SIZE 128
#define MAX_FREQ_BINS (NFFT / 2)
#define MAX_TIME_BINS ((NUM_SAMPLES - WINDOW_SIZE) / HOP_SIZE + 1)
#define MAX_BLOB_TIMES 7100
#define MAX_MIDPOINTS 100 // Adjust as needed based on expected number of midpoints
#define PI 3.141592653589793

// Function prototypes
bool butter_bandpass(double lowcut, double highcut, double *b, double *a);
void butter_bandpass_filter(double *data, int n, double *b, double *a, double *output);
void compute_spectrogram(double *signal, int signal_length, int fs,
                         double *frequencies, double *times, double Sxx[MAX_FREQ_BINS][MAX_TIME_BINS],
                         int *freq_bins, int *time_bins);
double sum_intense(double lower, double upper, double half_range, double *frequencies, int freq_bins,
                   double *times, int time_bins, double intensity_dB_filtered[MAX_FREQ_BINS][MAX_TIME_BINS],
                   double midpoint);
int find_midpoints(double *data, int num_frames, int samplingFreq, double *midpoints);

void setup()
{
  Serial.begin(9600);
}

void loop()
{
  // Replace this with actual audio data acquisition
  double data[NUM_SAMPLES];
  // Fill 'data' with your audio samples here

  int samplingFreq = 16000; // Adjust according to your sampling rate
  int num_frames = NUM_SAMPLES;

  // Apply Butterworth bandpass filter
  double filtered_signal[NUM_SAMPLES];
  double lowcut = 6000.0;
  double highcut = 15000.0;
  double b[9];
  double a[9];

  if (!butter_bandpass(lowcut, highcut, b, a))
  {
    // Handle error
    return;
  }

  butter_bandpass_filter(data, num_frames, b, a, filtered_signal);

  // Compute spectrogram
  double frequencies[MAX_FREQ_BINS];
  double times[MAX_TIME_BINS];
  double Sxx[MAX_FREQ_BINS][MAX_TIME_BINS];
  int freq_bins = 0, time_bins = 0;

  compute_spectrogram(filtered_signal, num_frames, samplingFreq, frequencies, times, Sxx, &freq_bins, &time_bins);

  // Continue with the rest of your processing...

  // Find midpoints
  double midpoints[MAX_MIDPOINTS];
  int num_midpoints = find_midpoints(data, num_frames, samplingFreq, midpoints);

  // Output the midpoints
  for (int i = 0; i < num_midpoints; i++)
  {
    Serial.print("Midpoint ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(midpoints[i]);
  }

  // Add the rest of your code here...

  // Since loop() runs repeatedly, you may want to add a delay or condition to prevent continuous execution
  while (1)
    ;
}

// Butterworth bandpass filter coefficients
bool butter_bandpass(double lowcut, double highcut, double *b, double *a)
{
  if (lowcut == 2000 && highcut == 6000)
  {
    b[0] = 0.00021314;
    b[1] = 0.;
    b[2] = -0.00085255;
    b[3] = 0.;
    b[4] = 0.00127883;
    b[5] = 0.;
    b[6] = -0.00085255;
    b[7] = 0.;
    b[8] = 0.00021314;

    a[0] = 1.;
    a[1] = -7.12847885;
    a[2] = 22.41882266;
    a[3] = -40.62891245;
    a[4] = 46.40780141;
    a[5] = -34.21333503;
    a[6] = 15.89913237;
    a[7] = -4.25840048;
    a[8] = 0.50337536;
  }
  else if (lowcut == 6000 && highcut == 15000)
  {
    b[0] = 0.00386952;
    b[1] = 0.;
    b[2] = -0.01547807;
    b[3] = 0.;
    b[4] = 0.02321711;
    b[5] = 0.;
    b[6] = -0.01547807;
    b[7] = 0.;
    b[8] = 0.00386952;

    a[0] = 1.;
    a[1] = -5.22664543;
    a[2] = 12.83819436;
    a[3] = -19.22549589;
    a[4] = 19.15517565;
    a[5] = -12.98213646;
    a[6] = 5.84957071;
    a[7] = -1.60753218;
    a[8] = 0.2088483;
  }
  else
  {
    Serial.println("Invalid bandpass range");
    return false;
  }
  return true;
}

// Butterworth bandpass filter application
void butter_bandpass_filter(double *data, int n, double *b, double *a, double *output)
{
  double w[9] = {0}; // State variables initialized to zero

  for (int i = 0; i < n; i++)
  {
    double w0 = data[i];
    for (int j = 1; j < 9; j++)
    {
      w0 -= a[j] * w[j - 1];
    }

    double y = b[0] * w0;
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

// Compute spectrogram using PlainFFT
void compute_spectrogram(double *signal, int signal_length, int fs,
                         double *frequencies, double *times, double Sxx[MAX_FREQ_BINS][MAX_TIME_BINS],
                         int *freq_bins, int *time_bins)
{
  int window_size = WINDOW_SIZE;
  int hop_size = HOP_SIZE;
  int nfft = NFFT;
  int num_freq_bins = nfft / 2;
  int num_time_bins = (signal_length - window_size) / hop_size + 1;

  *freq_bins = num_freq_bins;
  *time_bins = num_time_bins;

  // Create the window (Hanning window)
  double window[WINDOW_SIZE];
  for (int i = 0; i < window_size; i++)
  {
    window[i] = 0.5 * (1 - cos(2 * PI * i / (window_size - 1)));
  }

  // Compute frequency values
  for (int i = 0; i < num_freq_bins; i++)
  {
    frequencies[i] = (double)i * fs / nfft;
  }

  // Compute time values
  for (int t = 0; t < num_time_bins; t++)
  {
    int start = t * hop_size;
    times[t] = ((double)(start + window_size / 2)) / fs;
  }

  // Compute spectrogram
  double vReal[NFFT];
  double vImag[NFFT];

  for (int t = 0; t < num_time_bins; t++)
  {
    int start = t * hop_size;
    // Extract segment and apply window
    for (int i = 0; i < window_size; i++)
    {
      if (start + i < signal_length)
      {
        vReal[i] = signal[start + i] * window[i];
      }
      else
      {
        vReal[i] = 0.0;
      }
      vImag[i] = 0.0;
    }
    // Zero-pad if necessary
    for (int i = window_size; i < nfft; i++)
    {
      vReal[i] = 0.0;
      vImag[i] = 0.0;
    }

    // Perform FFT
    FFT.Compute(vReal, vImag, nfft, FFT_FORWARD);

    // Compute magnitude squared
    for (int f = 0; f < num_freq_bins; f++)
    {
      double mag_squared = vReal[f] * vReal[f] + vImag[f] * vImag[f];
      Sxx[f][t] = mag_squared;
    }
  }
}

// Sum intensity in a specific frequency and time range
double sum_intense(double lower, double upper, double half_range, double *frequencies, int freq_bins,
                   double *times, int time_bins, double intensity_dB_filtered[MAX_FREQ_BINS][MAX_TIME_BINS],
                   double midpoint)
{
  // Find frequency indices
  int freq_min_idx = 0;
  while (freq_min_idx < freq_bins && frequencies[freq_min_idx] < lower)
    freq_min_idx++;

  int freq_max_idx = freq_bins - 1;
  while (freq_max_idx >= 0 && frequencies[freq_max_idx] > upper)
    freq_max_idx--;

  // Ensure indices are within bounds
  if (freq_min_idx >= freq_bins)
    freq_min_idx = freq_bins - 1;
  if (freq_max_idx < 0)
    freq_max_idx = 0;

  // Swap if needed
  if (freq_min_idx > freq_max_idx)
  {
    int temp = freq_min_idx;
    freq_min_idx = freq_max_idx;
    freq_max_idx = temp;
  }

  // Find time indices
  int time_min_idx = 0;
  while (time_min_idx < time_bins && times[time_min_idx] < midpoint - half_range)
    time_min_idx++;

  int time_max_idx = time_bins - 1;
  while (time_max_idx >= 0 && times[time_max_idx] > midpoint + half_range)
    time_max_idx--;

  // Ensure indices are within bounds
  if (time_min_idx >= time_bins)
    time_min_idx = time_bins - 1;
  if (time_max_idx < 0)
    time_max_idx = 0;

  // Swap if needed
  if (time_min_idx > time_max_idx)
  {
    int temp = time_min_idx;
    time_min_idx = time_max_idx;
    time_max_idx = temp;
  }

  double total_intensity = 0.0;

  for (int i = freq_min_idx; i <= freq_max_idx; i++)
  {
    for (int j = time_min_idx; j <= time_max_idx; j++)
    {
      if (!isnan(intensity_dB_filtered[i][j]))
      {
        total_intensity += intensity_dB_filtered[i][j];
      }
    }
  }
  return total_intensity;
}

// Find midpoints in the signal
int find_midpoints(double *data, int num_frames, int samplingFreq, double *midpoints)
{
  const double lower_threshold_dB = 45.0;

  double lowcut = 2000.0;
  double highcut = 6000.0;
  double b[9];
  double a[9];
  if (!butter_bandpass(lowcut, highcut, b, a))
  {
    // Handle error
    return 0;
  }

  // Apply Butterworth bandpass filter
  double filtered_signal[NUM_SAMPLES];
  butter_bandpass_filter(data, num_frames, b, a, filtered_signal);

  // Compute spectrogram
  double frequencies[MAX_FREQ_BINS];
  double times[MAX_TIME_BINS];
  double Sxx[MAX_FREQ_BINS][MAX_TIME_BINS];
  int freq_bins = 0, time_bins = 0;

  compute_spectrogram(filtered_signal, num_frames, samplingFreq, frequencies, times, Sxx, &freq_bins, &time_bins);

  // Convert intensity to dB and apply threshold
  double intensity_dB_filtered[MAX_FREQ_BINS][MAX_TIME_BINS];
  bool valid_time_bins[MAX_TIME_BINS] = {false};

  for (int i = 0; i < freq_bins; i++)
  {
    for (int j = 0; j < time_bins; j++)
    {
      if (Sxx[i][j] > 0)
      {
        double intensity_dB = 10 * log10(Sxx[i][j] / 1e-12);
        if (intensity_dB > lower_threshold_dB)
        {
          intensity_dB_filtered[i][j] = intensity_dB;
          valid_time_bins[j] = true;
        }
        else
        {
          intensity_dB_filtered[i][j] = NAN;
        }
      }
      else
      {
        intensity_dB_filtered[i][j] = NAN;
      }
    }
  }

  // Collect blob_times where there is valid intensity
  double blob_times[MAX_BLOB_TIMES];
  int num_blob_times = 0;
  for (int j = 0; j < time_bins; j++)
  {
    if (valid_time_bins[j])
    {
      if (num_blob_times < MAX_BLOB_TIMES)
      {
        blob_times[num_blob_times++] = times[j];
      }
      else
      {
        // Exceeded maximum blob times
        break;
      }
    }
  }

  // Cluster blob_times
  const double time_tolerance = 0.05;    // seconds
  const double min_blob_duration = 0.15; // seconds

  int num_midpoints = 0;
  int cluster_start_idx = 0;
  while (cluster_start_idx < num_blob_times)
  {
    int cluster_end_idx = cluster_start_idx;
    while (cluster_end_idx + 1 < num_blob_times &&
           (blob_times[cluster_end_idx + 1] - blob_times[cluster_end_idx]) <= time_tolerance)
    {
      cluster_end_idx++;
    }

    double cluster_duration = blob_times[cluster_end_idx] - blob_times[cluster_start_idx];
    if (cluster_duration >= min_blob_duration)
    {
      // Calculate midpoint
      int cluster_size = cluster_end_idx - cluster_start_idx + 1;
      double sum_times = 0.0;
      for (int i = cluster_start_idx; i <= cluster_end_idx; i++)
      {
        sum_times += blob_times[i];
      }
      double midpoint = sum_times / cluster_size;
      if (num_midpoints < MAX_MIDPOINTS)
      {
        midpoints[num_midpoints++] = midpoint;
      }
      else
      {
        // Exceeded maximum midpoints
        break;
      }
    }

    cluster_start_idx = cluster_end_idx + 1;
  }

  return num_midpoints;
}
