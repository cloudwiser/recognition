# -------------------------------------------------
# recog_helpers.py
#
# Nick Hall : cloudwise.co
# 
# copyright cloudwise consulting 2019
# -------------------------------------------------

import sys
import cv2 as cv
import argparse
from os import mkdir
from os.path import isfile, exists
if sys.version_info.major == 3:
    from urllib.parse import urlparse   # Python 3.x
else:
    from urlparse import urlparse       # Python 2.7

# -------------------------------------------------

DEFAULT_SSD_MN1_THRESHOLD       = 0.4   # SSD MobileNet v1 confidence threshold
DEFAULT_SSD_MN2_THRESHOLD       = 0.4   # SSD MobileNet v2 confidence threshold - was 0.3
DEFAULT_YOLO3_THRESHOLD         = 0.75  # YOLO v3 confidence threshold
DEFAULT_FASTER_RCNN_THRESHOLD   = 0.4   # Faster RCNN confidence threshold

DEFAULT_SSD_MN1_CLASS = {15}            # Person from the SSD MobileNet v1 Caffe traing set   
DEFAULT_COCO_CLASS    = {1}             # Person from the COCO training set   
DEFAULT_YOLO3_CLASS   = {0}             # Person from the YOLOv3 training set   
DEFAULT_NMS_THRESHOLD = 0.4

SSD_MN1_MODEL   = "ssdmn1"
SSD_MN2_MODEL   = "ssdmn2"
YOLO3_MODEL     = "yolo3"
FASTER_RCNN_MODEL = "fasterrcnn"
# MASK_RCNN_MODEL = "maskrcnn"

# MASK_RCNN_MODEL_PATH = "./mask_rcnn_inception_v2_coco_2018_01_28/"
# MASK_RCNN_TEXT_GRAPH = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

SSD_MN1_MODEL_PATH = "./" + SSD_MN1_MODEL + "/"
SSD_MN1_TEXT_GRAPH = SSD_MN1_MODEL_PATH + "deploy.prototxt"
SSD_MN1_MODEL_WEIGHTS = SSD_MN1_MODEL_PATH + "mobilenet_iter_73000.caffemodel"

SSD_MN2_MODEL_PATH = "./" + SSD_MN2_MODEL + "/"
SSD_MN2_TEXT_GRAPH = SSD_MN2_MODEL_PATH + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
SSD_MN2_MODEL_WEIGHTS = SSD_MN2_MODEL_PATH + "frozen_inference_graph.pb"

YOLO3_MODEL_PATH = "./" + YOLO3_MODEL + "/"
YOLO3_TEXT_GRAPH = YOLO3_MODEL_PATH + "yolov3.cfg"
YOLO3_MODEL_WEIGHTS = YOLO3_MODEL_PATH + "yolov3.weights"

FASTER_RCNN_MODEL_PATH = "./" + FASTER_RCNN_MODEL + "/"
FASTER_RCNN_TEXT_GRAPH = FASTER_RCNN_MODEL_PATH + "faster_rcnn_resnet50_coco_2018_01_28.pbtxt"
FASTER_RCNN_MODEL_WEIGHTS = FASTER_RCNN_MODEL_PATH + "frozen_inference_graph.pb"

DEFAULT_OUTPUT_PATH = './out'

NO_FRAME_WAIT       = (10)
DEFAULT_POLL_WAIT   = (0)

# -------------------------------------------------

# Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to run the object recogniser')
    parser.add_argument('--video', help='path to video file')
    parser.add_argument('--stream', help='path to video stream')
    parser.add_argument('--out', help='path to output directory')
    parser.add_argument('--headless', help='disable X-server output', action='store_true')
    parser.add_argument('--showlabels', help='enable object labels', action='store_true')
    parser.add_argument('--blur', help='blur object region(s)', action='store_true')
    parser.add_argument('--threshold', help='set the detection threshold', type=float)
    parser.add_argument('--classes', help='[comma-delimited] list of COCO or YOLO classes', type=str)
    parser.add_argument('--model', help='CNN model : set to yolo3 or ssd]', type=str)
    parser.add_argument('--noframewait', help='wait time (secs) if no frame found', type=int)
    parser.add_argument('--interval', help='poll interval (secs)', type=int)

    args = parser.parse_args()

    _outpath = None
    _headless = False
    _showlabels = False
    _blur = False
    _threshold = DEFAULT_SSD_MN1_THRESHOLD
    _detect_classes = DEFAULT_SSD_MN1_CLASS
    _model = SSD_MN1_MODEL
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
    elif _model == SSD_MN1_MODEL:
        _threshold = DEFAULT_SSD_MN1_THRESHOLD
    else:
        _threshold = DEFAULT_SSD_MN2_THRESHOLD

    if (args.classes):
        _detect_classes = [int(item) for item in args.classes.split(',')]
        # Use list > set > list to remove any duplicate classes
        _detect_classes = list(set(_detect_classes))
    elif _model == YOLO3_MODEL:
        _detect_classes = DEFAULT_YOLO3_CLASS
    elif _model == SSD_MN1_MODEL:
        _detect_classes = DEFAULT_SSD_MN1_CLASS
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
