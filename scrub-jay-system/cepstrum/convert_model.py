# convert_model.py  (run only once in Python)
import joblib, skl2onnx
from skl2onnx.common.data_types import FloatTensorType

N_MFCC = 20                    # identical to training
feat_dim = N_MFCC * 2          # mean+std
model   = joblib.load("scrubjay_svm.joblib")

onnx = skl2onnx.convert_sklearn(
          model,
          initial_types=[('input', FloatTensorType([None, feat_dim]))],
          target_opset=17)

with open("scrubjay_svm.onnx", "wb") as f:
    f.write(onnx.SerializeToString())
print("→ scrubjay_svm.onnx created")
