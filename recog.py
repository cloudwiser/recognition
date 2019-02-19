# -------------------------------------------------
# recog : a OpenCV DNN-based Object Recogniser
# 
# With thanks to Satya Mallick & Sunita Nayak
# https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN
# https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
#
# Also see https://github.com/opencv/opencv/tree/master/samples/dnn
#
# Nick Hall : cloudwise.co
# 
# copyright cloudwise consulting 2019
# -------------------------------------------------

import sys
import cv2 as cv
import argparse
import numpy as np
from os import mkdir
from os.path import isfile, exists
import datetime
import time
if sys.version_info.major == 3:
    from urllib.parse import urlparse   # Python 3.x
else:
    from urlparse import urlparse       # Python 2.7

# -------------------------------------------------

COCO_classes = ["background", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
    "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
    "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

DEFAULT_SSD_THRESHOLD           = 0.8   # SSD confidence threshold - was 0.3
DEFAULT_YOLO3_THRESHOLD         = 0.5   # YOLO v3 confidence threshold
DEFAULT_FASTER_RCNN_THRESHOLD   = 0.4   # Faster RCNN confidence threshold

DEFAULT_COCO_CLASS    = {1}             # Person from the COCO set   
DEFAULT_YOLO3_CLASS   = {0}             # Person from the YOLOv3 set   
DEFAULT_NMS_THRESHOLD = 0.4

SSD_MODEL   = "ssdmn2"
YOLO3_MODEL = "yolo3"
FASTER_RCNN_MODEL = "fasterrcnn"
# MASK_RCNN_MODEL = "maskrcnn"

# MASK_RCNN_MODEL_PATH = "./mask_rcnn_inception_v2_coco_2018_01_28/"
# MASK_RCNN_TEXT_GRAPH = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
SSD_MODEL_PATH = "./" + SSD_MODEL + "/"
SSD_TEXT_GRAPH = SSD_MODEL_PATH + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
SSD_MODEL_WEIGHTS = SSD_MODEL_PATH + "frozen_inference_graph.pb"

YOLO3_MODEL_PATH = "./" + YOLO3_MODEL + "/"
YOLO3_TEXT_GRAPH = YOLO3_MODEL_PATH + "yolov3.cfg"
YOLO3_MODEL_WEIGHTS = YOLO3_MODEL_PATH + "yolov3.weights"

FASTER_RCNN_MODEL_PATH = "./" + FASTER_RCNN_MODEL + "/"
FASTER_RCNN_TEXT_GRAPH = FASTER_RCNN_MODEL_PATH + "faster_rcnn_resnet50_coco_2018_01_28.pbtxt"
FASTER_RCNN_MODEL_WEIGHTS = FASTER_RCNN_MODEL_PATH + "frozen_inference_graph.pb"

DEFAULT_OUTPUT_PATH = './out'
APP_NAME = 'recog : cloudwise.co : '

NO_FRAME_WAIT       = (10)
DEFAULT_POLL_WAIT   = (0)

CV_TEXT_SIZE        = 0.5
CV_BOUNDING_COLOR   = (255, 178, 50)

# -------------------------------------------------

# For each frame, draw a bounding box with optional label & blur for each detected-and-selected object:
# > SSD & Faster CNN
def objects_from_single_layer_output(frame, classes, detect_classes, predictions, threshold, showlabels, blur):
    _found = 0
    # num_classes = masks.shape[1]
    # num_detections = predictions.shape[2]
    frameH = frame.shape[0]
    frameW = frame.shape[1]

    # Obfuscate the frame for privacy?
    if blur:
        cv.GaussianBlur(frame, (23, 23), 30)

    for detection in predictions[0, 0, :, :]:
        score = detection[2]
        # Is this object confidence above the threshold
        if score > threshold:
            class_id = int(detection[1])

            # print("DEBUG: score={}, class={}".format(score, class_id))

            # Is this class one that we are interested in?
            if class_id not in detect_classes:
                continue
            else:
                _found += 1

            # Extract the bounding box
            left = int(frameW * detection[3])
            top = int(frameH * detection[4])
            right = int(frameW * detection[5])
            bottom = int(frameH * detection[6])
            
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Draw bounding box on the image
            cv.rectangle(frame, (left, top), (right, bottom), CV_BOUNDING_COLOR, 1)            

            # Show the object info?
            if showlabels:
                show_labels(frame, top, left, class_id, classes, score)
    return _found

# > YOLO v3 
def objects_from_multi_layer_output(frame, classes, detect_classes, predictions, threshold, showlabels, blur):
    _found = 0
    class_ids = []
    confidences = []
    boxes = []
    width = frame.shape[1]
    height = frame.shape[0]

    # Obfuscate the frame for privacy?
    if blur:
        cv.GaussianBlur(frame, (23, 23), 30)

    # For YOLO v3, we have multiple output layers as opposed to the single layer in SSD et al...
    for out in predictions:
        # For each detection found in the layer
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            score = scores[class_id]
            
            if score > threshold:
                
                # print("DEBUG: score={}, class={}".format(score, class_id))

                # Is this class one that we are interested in?
                if class_id not in detect_classes:
                    continue
                else:
                    _found += 1

                # Add the bounding box and score to the respective lists
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(score))
                boxes.append([x, y, w, h])
 
    # Perform maximal suppression on the (potentially-mulitple) bounding boxes per object
    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold, DEFAULT_NMS_THRESHOLD)

    # Display the bounding box on each object
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        left = int(round(x))
        top = int(round(y))
        right = int(round(x + w))
        bottom = int(round(y + h))
        cv.rectangle(frame, (left, top), (right, bottom), CV_BOUNDING_COLOR, 1)             

        # Show the object info?
        if showlabels:
            show_labels(frame, top, left, class_ids[i], classes, confidences[i])
    return _found

# Blur object regions for obfuscation
def blur_region(frame, top, bottom, left, right):
    region = frame[top:bottom, left:right]
    # apply a gaussian blur on the bounding region
    region = cv.GaussianBlur(region, (23, 23), 30)
    # merge this blurry rectangle into the frame
    frame[top:top + region.shape[0], left:left + region.shape[1]] = region
    return frame           

# Overlay object labels
def show_labels(frame, top, left, class_id, classes, score):
    # create the object label
    assert(class_id < len(classes))
    label = '%s:%.2f' % (classes[class_id], score)
    labelsize, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, CV_TEXT_SIZE, 1)
    labeltop = max(top, labelsize[1])
    top = int(labeltop - round(1.25*labelsize[1]))
    right = int(left + round(1.25*labelsize[0]))
    bottom = labeltop + baseline
    cv.rectangle(frame, (left, top), (right, bottom), CV_BOUNDING_COLOR, cv.FILLED)
    cv.putText(frame, label, (left, labeltop), cv.FONT_HERSHEY_SIMPLEX, CV_TEXT_SIZE, (0,0,0), 1)
    return frame           

# Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to run the object recogniser')
    parser.add_argument('--video', help='path to video file')
    parser.add_argument('--stream', help='path to video stream')
    parser.add_argument('--out', help='path to output directory')
    parser.add_argument('--headless', help='disable X-server output', action='store_true')
    parser.add_argument('--showlabels', help='enable object labels', action='store_true')
    parser.add_argument('--blur', help='blur object region(s)', action='store_true')
    parser.add_argument('--threshold', help='set the detection threshold', type=float, default=DEFAULT_SSD_THRESHOLD)
    parser.add_argument('--classes', help='[comma-delimited] list of COCO or YOLO classes', type=str)
    parser.add_argument('--model', help='CNN model : set to yolo3 or ssd]', type=str)
    parser.add_argument('--noframewait', help='wait time (secs) if no frame found', type=int)
    parser.add_argument('--interval', help='poll interval (secs)', type=int)

    args = parser.parse_args()

    _outpath = None
    _headless = False
    _showlabels = False
    _blur = False
    _threshold = DEFAULT_SSD_THRESHOLD
    _detect_classes = DEFAULT_COCO_CLASS
    _model = SSD_MODEL
    _noframewait = NO_FRAME_WAIT
    _interval = DEFAULT_POLL_WAIT

    if (args.model):
        _model = str.lower(args.model)

    if (args.showlabels):
        _showlabels = True

    if (args.headless):
        _headless = True

    if (args.blur):
        _blur = True

    if (args.noframewait):
        _noframewait = int(args.noframewait)

    if (args.interval):
        _interval = int(args.interval)

    if (args.threshold):
        _threshold = float(args.threshold)
    elif _model == YOLO3_MODEL:
        _threshold = DEFAULT_YOLO3_THRESHOLD
    elif _model == FASTER_RCNN_MODEL:
        _threshold = DEFAULT_FASTER_RCNN_THRESHOLD
    else:
        _threshold = DEFAULT_SSD_THRESHOLD

    if (args.classes):
        _detect_classes = [int(item) for item in args.classes.split(',')]
        # Use list > set > list to remove any duplicate classes
        _detect_classes = list(set(_detect_classes))
    elif _model == YOLO3_MODEL:
        _detect_classes = DEFAULT_YOLO3_CLASS
    else:
        _detect_classes = DEFAULT_COCO_CLASS

    if (args.out):
        # Get the output path for images
        if not exists(args.out):
            print("INFO: output path:", args.out, " doesn't exist...creating:", args.out)
            mkdir(args.out)
            if exists(args.out):
                _outpath = args.out
            else:
                print("ERR: can't create output path: ", args.out)
                sys.exit(1)
        else:
            _outpath = args.out

    # Parse the command line args for the capture source
    if (args.video):
        # Open a video file
        if not isfile(args.video):
            print("ERR: input video file: ", args.video, " doesn't exist")
            sys.exit(1)
        else:
            _capture = cv.VideoCapture(args.video)
    elif (args.stream):
        # Open a video stream
        if not urlparse(args.stream).scheme:
            print("ERR: input video stream: ", args.stream, " doesn't exist")
            sys.exit(1)
        else:
            _capture = cv.VideoCapture(args.stream)
    else:
        # ...or default to a local webcam stream
        _capture = cv.VideoCapture(0)
    return _capture, _outpath, _headless, _showlabels, _threshold, \
        _detect_classes, _blur, _model, _noframewait, _interval

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

# YOLO v3 net loader
def load_YOLO3_net(weights, config):
    # Load the network
    return cv.dnn.readNet(weights, config)

# YOLO v3 output layer retrieval
def get_YOLO3_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Get the SSD MobileNet candidate object boxes
def get_SSD_objects(region, net, net_params):
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

# -------------------------------------------------

if __name__ == "__main__":
    # Extract the various command line parameters
    capture, outpath, headless, showlabels, threshold, \
        detect_classes, blur, model, noframewait, interval = get_arguments()

    # Load the relevant classes and model - default to SSD MobileNet
    if model == YOLO3_MODEL:
        classes = load_YOLO3_classes(YOLO3_MODEL_PATH + "yolov3.classes")
        net = load_YOLO3_net(YOLO3_MODEL_WEIGHTS, YOLO3_TEXT_GRAPH)
    elif model == FASTER_RCNN_MODEL:
        classes = load_COCO_classes(FASTER_RCNN_MODEL_PATH + "mscoco_labels.names")
        net = load_TF_net(FASTER_RCNN_MODEL_WEIGHTS, FASTER_RCNN_TEXT_GRAPH)
    else:
        classes = load_COCO_classes(SSD_MODEL_PATH + "mscoco_labels.names")
        net = load_TF_net(SSD_MODEL_WEIGHTS, SSD_TEXT_GRAPH)

    # Set the output window name (assuming there is a GUI output path)
    if not headless:
        cv.namedWindow(APP_NAME, cv.WINDOW_NORMAL)

    # Frame processing loop
    print("INFO: frame acquisition...")
    while True:
        # Get a frame from the video/image/stream
        hasFrame, frame = capture.read()
        
        # Skip and sleep if there is no frame
        if not hasFrame:
            print("WARN: no frame...waiting {} sec(s)".format(noframewait))
            time.sleep(noframewait)
            continue
        else:
            height, width = frame.shape[:2]

        # Get the object predictions and annotate the frame - default to SSD MobileNet
        if model == YOLO3_MODEL:
            predictions = get_YOLO3_objects(frame, net, get_YOLO3_output_layers(net))
            found = objects_from_multi_layer_output(frame, classes, detect_classes, predictions, threshold, showlabels, blur)
        # elif model == MASK_RCNN_MODEL:
        #   predictions, masks = get_mask_RCNN_objects(frame, net, ['detection_out_final', 'detection_masks'])
        elif model == FASTER_RCNN_MODEL:
            predictions = get_Faster_RCNN_objects(frame, net, None)
            found = objects_from_single_layer_output(frame, classes, detect_classes, predictions, threshold, showlabels, blur)
        else:   # SSD model
            predictions = get_SSD_objects(frame, net, None)
            found = objects_from_single_layer_output(frame, classes, detect_classes, predictions, threshold, showlabels, blur)

        # Watermark the frame
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        t, _ = net.getPerfProfile()
        performance = ' : infer=%0.0fms' % abs(t * 1000.0 / cv.getTickFrequency())
        modelused = ' : ' + model + '@' + str(width) + 'x' + str(height)
        label = APP_NAME + timestamp + performance + modelused
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, CV_TEXT_SIZE, (0, 0, 0), 1)

        # Write the frame to output directory
        if found > 0 and outpath:
            outputFile = outpath + '/' + timestamp + '.jpg'
            cv.imwrite(outputFile, frame.astype(np.uint8))

        # Display the frame to X if there is a GUI path
        if not headless:
            cv.imshow(APP_NAME, frame)
        
        # Esc to quit
        if not headless and cv.waitKey(1) == 27: 
            frame.release()
            break

        # Wait
        time.sleep(interval)

print("INF: stopped frame processing")
if not headless:
    cv.destroyAllWindows()
