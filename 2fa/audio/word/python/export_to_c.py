import numpy as np
from tensorflow import keras

MODEL_PATH = "models/audio_classifier_model.keras"
SCALER_MEAN_PATH = "models/scaler_mean.npy"
SCALER_SCALE_PATH = "models/scaler_scale.npy"
OUTPUT_HEADER = "../c/model_params.h"

def write_array(f, c_type, name, arr, per_line=8):
    arr = np.asarray(arr, dtype=np.float32).flatten()
    f.write(f"static const {c_type} {name}[{arr.size}] = {{\n    ")
    for i, v in enumerate(arr):
        f.write(f"{float(v):.8e}f")
        if i != arr.size - 1:
            f.write(", ")
        if (i + 1) % per_line == 0:
            f.write("\n    ")
    f.write("\n};\n\n")

def main():
    print("Loading model and scaler...")
    model = keras.models.load_model(MODEL_PATH)
    scaler_mean = np.load(SCALER_MEAN_PATH).astype(np.float32)
    scaler_scale = np.load(SCALER_SCALE_PATH).astype(np.float32)

    input_size = scaler_mean.size  # should be 13 * 1000 = 13000

    # Get dense layers in order
    dense_layers = [l for l in model.layers if "dense" in l.name]
    if len(dense_layers) != 4:
        raise RuntimeError(f"Expected 4 Dense layers, found {len(dense_layers)}")
    dense1, dense2, dense3, dense4 = dense_layers

    k1, b1 = dense1.get_weights()
    k2, b2 = dense2.get_weights()
    k3, b3 = dense3.get_weights()
    k4, b4 = dense4.get_weights()

    with open(OUTPUT_HEADER, "w") as f:
        f.write("#ifndef MODEL_PARAMS_H\n")
        f.write("#define MODEL_PARAMS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"#define INPUT_SIZE {input_size}\n")
        f.write(f"#define DENSE1_UNITS {k1.shape[1]}\n")
        f.write(f"#define DENSE2_UNITS {k2.shape[1]}\n")
        f.write(f"#define DENSE3_UNITS {k3.shape[1]}\n")
        f.write(f"#define DENSE4_UNITS {k4.shape[1]}\n\n")

        write_array(f, "float", "SCALER_MEAN", scaler_mean)
        write_array(f, "float", "SCALER_SCALE", scaler_scale)

        write_array(f, "float", "DENSE1_KERNEL", k1)
        write_array(f, "float", "DENSE1_BIAS",   b1)

        write_array(f, "float", "DENSE2_KERNEL", k2)
        write_array(f, "float", "DENSE2_BIAS",   b2)

        write_array(f, "float", "DENSE3_KERNEL", k3)
        write_array(f, "float", "DENSE3_BIAS",   b3)

        write_array(f, "float", "DENSE4_KERNEL", k4)
        write_array(f, "float", "DENSE4_BIAS",   b4)

        f.write("#endif // MODEL_PARAMS_H\n")

    print(f"Wrote {OUTPUT_HEADER}")

if __name__ == "__main__":
    main()
