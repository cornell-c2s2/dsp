import wave
import struct

def pcm_to_wav(pcm_file, wav_file, num_channels=1, sample_rate=44100, sample_width=2):
    
    with open(pcm_file, 'rb') as pcm:
        pcm_data = pcm.read()

    with wave.open(wav_file, 'wb') as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)

if __name__ == "__main__":
    pcm_file = 'sample_audio.pcm'
    wav_file = 'output.wav'
    
    #num_channels is 1 for Mono and 2 for Stereo
    #sample_rate in kHz
    #sample_width is number of bytes per sample (i.e. 3 for 24-bit audio)
    pcm_to_wav(pcm_file, wav_file, num_channels=2, sample_rate=96000, sample_width=3)
