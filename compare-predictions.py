import numpy as np
tfOut = np.load("output/yolov2-predictions.npy")
onnxOut = np.load("output/onnx-predictions.npy")

print("Comparing model predictions!")

# Comparing the arrays
if np.array_equal(tfOut, onnxOut):
    print("Equal")
else:
    print("Not Equal")
