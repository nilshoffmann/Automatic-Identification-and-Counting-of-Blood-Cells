from pathlib import Path
import distutils
import time
from utils import iou
from scipy import spatial
from darkflow.net.build import TFNet
import cv2
import numpy as np
import tensorflow as tf
from darkflow.utils.im_transform import imcv2_affine_trans, imcv2_recolor
import onnx
import onnxruntime as rt
import tf2onnx
import pickle

outDir = Path('output')
outDir.mkdir(parents=True, exist_ok=True)
tf.compat.v1.disable_eager_execution()
options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.1,
           'gpu': 0.7}

tfnet = TFNet(options)

pred_bb = []  # predicted bounding box
pred_cls = []  # predicted class
pred_conf = []  # predicted class confidence

def blood_cell_count(tfnet, file_name):
    rbc = 0
    wbc = 0
    platelets = 0

    cell = []
    cls = []
    conf = []

    record = []
    tl_ = []
    br_ = []
    iou_ = []
    iou_value = 0

    tic = time.time()
    image = cv2.imread('data/' + file_name)
    # save dictionary to person_data.pkl file
    with open('output/meta.pkl', 'wb') as fp:
        pickle.dump(tfnet.meta, fp)
        print('Saved meta information to dictionary: output/meta.pkl')
    
    output = tfnet.return_predict(image)
    print("Saving TF Model Predictions in 'yolov2-predictions.npy'")

    for prediction in output:
        print(prediction)
        label = prediction['label']
        confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        if label == 'RBC' and confidence < .5:
            continue
        if label == 'WBC' and confidence < .25:
            continue
        if label == 'Platelets' and confidence < .25:
            continue

        # clearing up overlapped same platelets
        if label == 'Platelets':
            if record:
                tree = spatial.cKDTree(record)
                index = tree.query(tl)[1]
                iou_value = iou(tl + br, tl_[index] + br_[index])
                iou_.append(iou_value)

            if iou_value > 0.1:
                continue

            record.append(tl)
            tl_.append(tl)
            br_.append(br)

        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        center = (center_x, center_y)

        if label == 'RBC':
            color = (255, 0, 0)
            rbc = rbc + 1
        if label == 'WBC':
            color = (0, 255, 0)
            wbc = wbc + 1
        if label == 'Platelets':
            color = (0, 0, 255)
            platelets = platelets + 1

        radius = int((br[0] - tl[0]) / 2)
        image = cv2.circle(image, center, radius, color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, label, (center_x - 15, center_y + 5), font, .5, color, 1)
        cell.append([tl[0], tl[1], br[0], br[1]])

        if label == 'RBC':
            cls.append(0)
        if label == 'WBC':
            cls.append(1)
        if label == 'Platelets':
            cls.append(2)

        conf.append(confidence)

    toc = time.time()
    pred_bb.append(cell)
    pred_cls.append(cls)
    pred_conf.append(conf)
    avg_time = (toc - tic) * 1000
    print('{0:.5}'.format(avg_time), 'ms')

    cv2.imwrite('output/' + file_name, image)

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

# Evaluate original method and model and onnx model predictions
image_name = 'image_001.jpg'
blood_cell_count(tfnet, image_name)
print('Saving TF model!')
tfnet.savepb()

distutils.dir_util.copy_tree('built_graph', 'output')

name = 'built_graph/tiny-yolo-voc-3c.pb'
print(f"Parsing Model {name}")
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
placeholders = [ op for op in graph.get_operations() if op.type == "Placeholder"]
print("Placeholders:")
print(placeholders)

# saving the model
tf.compat.v1.reset_default_graph()
tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    # inputs_as_nchw are optional, but with ONNX in NCHW and Tensorflow in NHWC format, it is best to add this option
    g = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=tinputs, output_names=toutputs,
                                        inputs_as_nchw=tinputs)

    model_proto = g.make_model('model_out')
    checker = onnx.checker.check_model(model_proto)
    # do not include a defined value for placeholder via feed_dict, we want to store the model as is, not with any set placeholders
    tf2onnx.utils.save_onnx_model("./output/", "saved_model", feed_dict={}, model_proto=model_proto)

print("Saved ONNX model to saved_model.onnx")
