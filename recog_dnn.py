# -------------------------------------------------
# recog_dnn.py
#
# Nick Hall : cloudwise.co
# 
# copyright cloudwise consulting 2019
# -------------------------------------------------

import sys
import cv2 as cv

# -------------------------------------------------

# COCO classes loader
def load_COCO_classes(classes_file_path):
    # Load names of COCO classes
    _classes = None
    with open(classes_file_path, 'rt') as f:
        _classes = f.read().rstrip('\n').split('\n')
    return _classes

# YOLO v3 classes loader
def load_YOLO3_classes(classes_file_path):
    classes = None
    with open(classes_file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# TF net loader
def load_TF_net(model_weights, text_graph):
    # Load the network
    _net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
    _net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    _net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return _net

# Caffe net loader
def load_Caffe_net(model_weights, text_graph):
    return cv.dnn.readNetFromCaffe(text_graph, model_weights)

# YOLO v3 net loader
def load_YOLO3_net(model_weights, text_graph):
    # Load the network
    return cv.dnn.readNet(model_weights, text_graph)

# YOLO v3 output layer retrieval
def get_YOLO3_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Get the SSD MobileNet v1 candidate object boxes
def get_SSD_MobileNet1_objects(region, net, net_params):
    height, width = region.shape[:2]
    
    # Create a 4D blob from the region
    blob = cv.dnn.blobFromImage(region, 0.007843, (width, height), 127.5)
    net.setInput(blob)
    
    # Run the forward pass to get object boxes from the output layers
    return net.forward(net_params)

# Get the SSD MobileNet v2 candidate object boxes
def get_SSD_MobileNet2_objects(region, net, net_params):
    # resize frame for prediction
    # region_resized = cv.resize(region, (300,300))
    
    # Create a 4D blob from the region
    blob = cv.dnn.blobFromImage(region, swapRB=True, crop=False)
    # blob = cv.dnn.blobFromImage(region_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)    
    net.setInput(blob)
    
    # Run the forward pass to get object boxes from the output layers
    return net.forward(net_params)

# Get the YOLO v3 candidate object boxes
def get_YOLO3_objects(region, net, net_params):
    # Create a 4D blob from the region
    # blob = cv.dnn.blobFromImage(region, scale, size=(416, 416), mean=(0,0,0), swapRB=False, crop=False)
    blob = cv.dnn.blobFromImage(region, 0.00392, (544, 544), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Run the forward pass to get object boxes from the output layers
    return net.forward(net_params)

# Get the Faster RCNN ResNet50 candidate object boxes
def get_Faster_RCNN_objects(region, net, net_params):
    # Create a 4D blob from the region
    blob = cv.dnn.blobFromImage(region, (300, 300), swapRB=True, crop=False)
    # blob = cv.dnn.blobFromImage(region, size=(300, 300), mean=(103.939, 116.779, 123.68), swapRB=False, crop=False)    
    net.setInput(blob)
    
    # Run the forward pass to get object boxes from the output layers
    return net.forward(net_params)
