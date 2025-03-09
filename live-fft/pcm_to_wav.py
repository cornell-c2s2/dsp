import numpy as np
import wave
import struct
import argparse

def pcm_to_wav(output_file, sample_rate=16000):
    """
    Convert normalized PCM samples (-1 to 1) to a WAV file.
    
    Args:
        output_file (str): Path to output WAV file
        sample_rate (int): Sample rate in Hz (default: 16000)
    """
    input_file = "birdQ.txt"
    
    # Read the normalized PCM data
    with open(input_file, "r") as file:
        content = file.read().strip()
        pcm_data = np.array([float(num) for num in content.split(",")])
    
    # Scale from -1,1 to -32768,32767 (16-bit range)
    scaled_data = np.int16(pcm_data * 32767)
    
    # Create WAV file
    with wave.open(output_file, 'wb') as wav_file:
        # Set parameters: nchannels, sampwidth, framerate, nframes, comptype, compname
        wav_file.setparams((1, 2, sample_rate, len(scaled_data), 'NONE', 'not compressed'))
        
        # Write frames
        wav_file.writeframes(scaled_data.tobytes())
    
    print(f"Successfully converted {input_file} to {output_file}")
    print(f"- Sample rate: {sample_rate} Hz")
    print(f"- Bit depth: 16-bit")
    print(f"- Channels: 1 (mono)")
    print(f"- Total samples: {len(pcm_data)}")
    print(f"- Duration: {len(pcm_data)/sample_rate:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert normalized PCM samples to WAV file.")
    parser.add_argument("output_file", help="Path to output WAV file")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    
    args = parser.parse_args()
    
    pcm_to_wav(args.output_file, args.sample_rate)