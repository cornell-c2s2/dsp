import numpy as np
from tensorflow import keras

model = keras.models.load_model("models/audio_classifier_model.keras")

weights = model.get_weights()
for i, w in enumerate(weights):
    print(i, w.shape)

def dump_c_array(name, arr):
    flat = arr.flatten()
    print(f"static const float {name}[{flat.size}] = {{")
    for i, v in enumerate(flat):
        end = "," if i < flat.size - 1 else ""
        print(f"    {v:.8e}{end}")
    print("};\n")

# dump_c_array("dense1_kernel", weights[0])
# dump_c_array("dense1_bias",   weights[1])
# dump_c_array("dense2_kernel", weights[2])
# dump_c_array("dense2_bias",   weights[3])
# dump_c_array("dense3_kernel", weights[4])
# dump_c_array("dense3_bias",   weights[5])
# dump_c_array("dense4_kernel", weights[6])
# dump_c_array("dense4_bias",   weights[7])

# Scaler params
import numpy as np
mean  = np.load("models/scaler_mean.npy")
scale = np.load("models/scaler_scale.npy")
# dump_c_array("scaler_mean", mean)
# dump_c_array("scaler_scale", scale)
