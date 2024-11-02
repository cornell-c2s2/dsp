import struct

def pcm_to_floats(pcm_file, sample_width=2):
    with open(pcm_file, 'rb') as pcm:
        pcm_data = pcm.read()

    if sample_width == 2:
        fmt = '<h' 
        max_int_value = 32768.0
        num_samples = len(pcm_data) // sample_width
        int_data = struct.unpack(fmt * num_samples, pcm_data)
    elif sample_width == 1:
        fmt = '<b'
        max_int_value = 128.0
        num_samples = len(pcm_data) // sample_width
        int_data = struct.unpack(fmt * num_samples, pcm_data)
    elif sample_width == 3:
        max_int_value = 8388608.0  # 2^23, as the range is -2^23 to 2^23-1
        num_samples = len(pcm_data) // sample_width
        int_data = []
        
        for i in range(num_samples):
            sample_bytes = pcm_data[i*3:(i*3)+3]
            sample_int = int.from_bytes(sample_bytes, byteorder='little', signed=True)
            int_data.append(sample_int)
    else:
        raise ValueError("Unsupported sample width")

    float_data = [sample / max_int_value for sample in int_data]

    normalized_float_data = [(sample + 1.0) / 2.0 for sample in float_data]

    return normalized_float_data

def write_floats_to_txt(float_data, txt_file):
    with open(txt_file, 'w') as f:
        for value in float_data:
            f.write(f"{value}\n")

if __name__ == "__main__":
    pcm_file = 'sample_audio.pcm'
    txt_file = 'sample_audio.txt'
    
    float_data = pcm_to_floats(pcm_file, sample_width=3)
    
    write_floats_to_txt(float_data, txt_file)
    
    print(f"Float data written to {txt_file}")
