from pathlib import Path
import pickle
import time
from utils import iou
from scipy import spatial
from darkflow.net.build import TFNet
import cv2
import numpy as np
import onnx
import onnxruntime as rt
from darkflow.utils.im_transform import imcv2_affine_trans, imcv2_recolor
from darkflow.utils.box import BoundBox
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

options = {'model': 'output/saved_model.onnx',
           'threshold': 0.1
          }

def findboxes(meta, net_out):
    # meta
    meta = meta
    boxes = list()
    boxes = box_constructor(meta, net_out)
    return boxes

def process_box(meta, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = meta['labels'][max_indx]
    
    print("x=%.2f w=%.2f y=%.2f h=%.2f"%(b.x, b.w, b.y, b.h))
    if max_prob > threshold:
        left = int((b.y - b.h / 2.) * w)
        right = int((b.y + b.h / 2.) * w)
        top = int((b.x - b.w / 2.) * h)
        bot = int((b.x + b.w / 2.) * h)
        if left < 0:  left = 0
        if right > w - 1: right = w - 1
        if top < 0:   top = 0
        if bot > h - 1:   bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None

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
    imsz = cv2.resize(im, (int(meta['inp_size'][0]), int(meta['inp_size'][1])))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return imsz

def return_predict_onnx(meta, onnx_model_file_name, im, options):
    assert isinstance(im, np.ndarray), \
        'Image is not a np.ndarray'
    h, w, _ = im.shape
    image = resize_input(im, meta)
    print(image.shape)
    # we need to adjust shapes, image has (416, 416, 3) while onnx inference session expects (3, 416, 416)
    this_inp = np.expand_dims(np.moveaxis(image, [2], [0]), 0)

    session = rt.InferenceSession(onnx_model_file_name, providers=rt.get_available_providers())
    input_name = session.get_inputs()[0].name
    input_names = [input.name for input in session.get_inputs()]
    print(input_names)
    output_names = [output.name for output in session.get_outputs()]
    print(output_names)
    onnx_out = session.run(output_names, {input_name: this_inp.astype(np.float32)})
    for t in session.get_inputs():
        print("Session input:", t.name, t.type, t.shape)

    for t in session.get_outputs():
        print("Session output:", t.name, t.type, t.shape)
    print("Saving ONNX Model Predictions in 'output/onnx-predictions.npy'")
    #print(outputs[0].shape, type(outputs))
    #print(pred_onnx)
    np.save('output/onnx-predictions', onnx_out)
    #im = tfnet.framework.resize_input(im)
    print(len(onnx_out))
    print(onnx_out[0].shape)
    boxes = findboxes(meta=meta, net_out=np.squeeze(onnx_out[0]))
    threshold = options['threshold']
    boxesInfo = list()
    for box in boxes:
        print(box)
        tmpBox = process_box(meta, box, w, h, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

def blood_cell_count_onnx(file_name, meta_file, options):

    onnx_model_file_name = options['model']

    with open(meta_file, 'rb') as handle:
        meta = pickle.load(handle)

    print("Model Meta: ")
    print(meta)

    print("Loading ONNX model from saved_model.onnx")
    # onnx_model is an in-memory ModelProto
    onnx_model = onnx.load(onnx_model_file_name)

    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid!")

    #print(onnx.helper.printable_graph(onnx_model.graph))

    pred_bb = []  # predicted bounding box
    pred_cls = []  # predicted class
    pred_conf = []

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

    # read test image
    print("Reading test image")
    tic = time.time()
    image = cv2.imread('data/' + file_name)
    
    output = return_predict_onnx(meta=meta, onnx_model_file_name='output/saved_model.onnx', im=image, options=options)

    timg = image#cv2.transpose(image)#cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_180)
    for prediction in output:
        print(prediction)
        label = prediction['label']
        confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        if label == 'RBC' and confidence < .5:
            print("Skipping RBC")
            continue
        if label == 'WBC' and confidence < .25:
            print("Skipping WBC")
            continue
        if label == 'Platelets' and confidence < .25:
            print("Skipping platelet")
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

        center_y = int((tl[0] + br[0]) / 2)
        center_x = int((tl[1] + br[1]) / 2)
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
        print("Label: %s, Center: x=%.2f y=%.2f, Radius: %.2f" % (label, center[0], center[1], radius))
        timg = cv2.circle(timg, center, radius, color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        timg = cv2.putText(timg, label, (center_x - 15, center_y + 5), font, .5, color, 1)
        cell.append([tl[1], tl[0], br[1], br[0]])

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

    cv2.imwrite('output/onnx-' + file_name, timg)

# Evaluate original method and model and onnx model predictions

blood_cell_count_onnx('image_001.jpg', 'output/meta.pkl', options)