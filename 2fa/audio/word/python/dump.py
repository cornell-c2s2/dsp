import wave
import numpy as np

#path = "../../data/testing/stop_121417.wav"
#path = "../../data/testing/bed__common_voice_en_82827.wav"
#path = "../../data/pos_stop.wav"
path = "../../data/gmm_test/pos_2.wav"
with wave.open(path, "rb") as w:
    assert w.getsampwidth() == 2
    assert w.getnchannels() in (1, 2)
    sr = w.getframerate()
    frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16)

    if w.getnchannels() == 2:
        data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

print("Sample rate:", sr, "num samples:", len(data))

with open("non_stop_clip.h", "w") as f:
    f.write("#ifndef STOP_CLIP_H\n#define STOP_CLIP_H\n\n")
    f.write(f"#define STOP_CLIP_NUM_SAMPLES {len(data)}\n\n")
    f.write("static const float STOP_CLIP[STOP_CLIP_NUM_SAMPLES] = {\n")
    for i, v in enumerate(data):
        f.write(f"{float(v/32768.0)}")
        if i != len(data) - 1:
            f.write(",")
        if (i + 1) % 16 == 0:
            f.write("\n")
    f.write("};\n\n#endif\n")
