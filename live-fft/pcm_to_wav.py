import numpy as np
from scipy.signal import resample_poly
from scipy.io import wavfile

INPUT_FILE = "bird_control.txt"
OUTPUT_WAV = "new_out.wav"

def convert_10k_to_16k_mono_to_stereo(input_file, output_wav):
    # Original sample rate
    fs_in = 10000  
    # Desired sample rate
    fs_out = 16000
    
    # 1) Read PCM samples from text file
    #    (Assuming one sample per line in birdQ.txt)
    with open(INPUT_FILE, "r") as file:
     content = file.read().strip()
     samples = np.array([float(num) for num in content.split(",")])
    
    # 2) Resample from 10 kHz to 16 kHz
    #    resample_poly(x, up, down) effectively does:
    #        new_length = len(x) * up / down
    #    Here up=16, down=10
    scaled_data = samples * 32768.0
    # with open("output1.txt", "w") as file:
    #     file.write(",".join(map(str, scaled_data)))
    # resampled = resample_poly(scaled_data, 16, 8)
    # with open("output2.txt", "w") as file:
    #     file.write(",".join(map(str, resampled)))
    
    # 3) Convert mono to stereo by duplicating the channel
    stereo_data = np.column_stack((scaled_data, scaled_data))
    
    # 4) Write to a 16-bit stereo WAV file at 16 kHz
    #    Make sure data type is int16 for standard PCM
    wavfile.write(output_wav, fs_out, stereo_data.astype(np.int16))

if __name__ == "__main__":
    convert_10k_to_16k_mono_to_stereo(INPUT_FILE, OUTPUT_WAV)
    print(f"Saved stereo .wav at 16 kHz to: {OUTPUT_WAV}")



# import wave
# import numpy as np


# input_file = "birdQ.txt"
    
# # Read the normalized PCM data
# with open(input_file, "r") as file:
#     content = file.read().strip()
#     adc_values = np.array([float(num) for num in content.split(",")])
# #adc_values = [] # add pcm from arduino code here
# pcm_values = np.array(adc_values).astype(np.int16)

# sampling_rate = 10000
# num_channels = 1
# sample_width = 2

# with wave.open("output_16KHz.wav", "wb") as wav_file:
#     wav_file.setnchannels(num_channels)
#     wav_file.setsampwidth(sample_width)
#     wav_file.setframerate(sampling_rate)
#     wav_file.writeframes(pcm_values.tobytes())





# import numpy as np
# import wave
# import struct
# import argparse

# def pcm_to_wav(output_file, sample_rate=16000):
#     """
#     Convert normalized PCM samples (-1 to 1) to a WAV file.
    
#     Args:
#         output_file (str): Path to output WAV file
#         sample_rate (int): Sample rate in Hz (default: 16000)
#     """
#     input_file = "birdQ.txt"
    
#     # Read the normalized PCM data
#     with open(input_file, "r") as file:
#         content = file.read().strip()
#         pcm_data = np.array([float(num) for num in content.split(",")])
    
#     # Scale from -1,1 to -32768,32767 (16-bit range)
#     scaled_data = np.int16(pcm_data * 32767)
    
#     # Create WAV file
#     with wave.open(output_file, 'wb') as wav_file:
#         # Set parameters: nchannels, sampwidth, framerate, nframes, comptype, compname
#         wav_file.setparams((1, 2, sample_rate, len(scaled_data), 'NONE', 'not compressed'))
        
#         # Write frames
#         wav_file.writeframes(scaled_data.tobytes())
    
#     print(f"Successfully converted {input_file} to {output_file}")
#     print(f"- Sample rate: {sample_rate} Hz")
#     print(f"- Bit depth: 16-bit")
#     print(f"- Channels: 1 (mono)")
#     print(f"- Total samples: {len(pcm_data)}")
#     print(f"- Duration: {len(pcm_data)/sample_rate:.2f} seconds")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert normalized PCM samples to WAV file.")
#     parser.add_argument("output_file", help="Path to output WAV file")
#     parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    
#     args = parser.parse_args()
    
#     pcm_to_wav(args.output_file, args.sample_rate)