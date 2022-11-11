import os

import cv2
import numpy as np
import tensorflow as tf
import onnx
import tf2onnx
import onnxruntime as rt
from onnx import helper, shape_inference
from onnx import TensorProto

from darkflow.net import yolov2
from darkflow.net.yolo.predict import _fix
from darkflow.utils.im_transform import imcv2_affine_trans, imcv2_recolor


def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    return (inputs, outputs)

def resize_input(im, meta):
    imsz = cv2.resize(im, (int(meta['inp_size']['w']), int(meta['inp_size']['h'])))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz


name = 'built_graph/tiny-yolo-voc-3c.pb'

graph_def = tf.compat.v1.GraphDef()
with open(name, 'rb') as f:
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

# optimize and save to ONNX
# Note: tf appends :0 to layer names
[inputs, outputs] = analyze_inputs_outputs(graph)
tinputs = ["input:0"] #inputs #[str(i) + ":0" for i in inputs]
toutputs = ["output:0"] #outputs #[str(o) + ":0" for o in outputs]

# saving the model
tf.compat.v1.reset_default_graph()
tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    # inputs_as_nchw are optional, but with ONNX in NCHW and Tensorflow in NHWC format, it is best to add this option
    g = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=tinputs, output_names=toutputs,
                                        inputs_as_nchw=tinputs)

    model_proto = g.make_model('model_out')
    checker = onnx.checker.check_model(model_proto)

    tf2onnx.utils.save_onnx_model("./", "saved_model", feed_dict={}, model_proto=model_proto)

print("Saved ONNX model to saved_model.onnx")

print("Loading ONNX model from saved_model.onnx")
# onnx_model is an in-memory ModelProto
onnx_model = onnx.load("saved_model.onnx")

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
else:
    print("The model is valid!")

print(onnx.helper.printable_graph(onnx_model.graph))

# read test image
print("Reading test image")
image = cv2.imread('data/' + 'image_001.jpg')
result = imcv2_affine_trans(image)
im, dims, trans_param = result
scale, offs, flip = trans_param
im = imcv2_recolor(im)
#output = tfnet.return_predict(image)
h, w, _ = im.shape
meta = {'inp_size': { 'h': 416, 'w': 416, 'c': 3} }
image = resize_input(im, meta)
this_inp = np.expand_dims(np.transpose(image), 0)

sess = rt.InferenceSession(
    "saved_model.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: this_inp.astype(np.float32)})[0]
for t in sess.get_inputs():
    print("Session input:", t.name, t.type, t.shape)

for t in sess.get_outputs():
    print("Session output:", t.name, t.type, t.shape)
print("Model Predictions for test image:")
#print(outputs[0].shape, type(outputs))
#print(pred_onnx)
np.save('onnx-predictions', pred_onnx)
