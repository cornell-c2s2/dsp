import wave
import struct

def wav_to_pcm(wav_file, pcm_file):
    with wave.open(wav_file, 'rb') as wav:
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()

        #print(f"Channels: {num_channels}")
        #print(f"Sample Width: {sample_width} bytes")
        #print(f"Sample Rate: {sample_rate} Hz")
        #print(f"Number of Frames: {num_frames}")
        frames = wav.readframes(num_frames)
    with open(pcm_file, 'wb') as pcm:
        pcm.write(frames)

if __name__ == "__main__":
    wav_file = 'sample_audio.WAV'
    pcm_file = 'sample_audio.pcm'
    
    wav_to_pcm(wav_file, pcm_file)