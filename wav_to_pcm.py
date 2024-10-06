import wave
import struct

def wav_to_pcm(wav_file, pcm_file):
    # Open the WAV file
    with wave.open(wav_file, 'rb') as wav:
        # Get parameters from the wav file
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()

        print(f"Channels: {num_channels}")
        print(f"Sample Width: {sample_width} bytes")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Number of Frames: {num_frames}")

        # Read all frames from the WAV file
        frames = wav.readframes(num_frames)

    # Write raw PCM data to a file
    with open(pcm_file, 'wb') as pcm:
        # Optionally, you could manipulate the frames here
        pcm.write(frames)

if __name__ == "__main__":
    wav_file = 'sample_audio.wav'  # Input .wav file
    pcm_file = 'sample_audio.pcm' # Output .pcm file
    
    wav_to_pcm(wav_file, pcm_file)