# -------------------------------------------------
# recog_config : config file parser helper function(s)
#
# Nick Hall
# 
# Copyright (c) 2019 cloudwise (http://cloudwise.co)
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
from configparser import ConfigParser, ExtendedInterpolation

# -------------------------------------------------
# Future models to implement...

# DEFAULT_FASTER_RCNN_THRESHOLD = 0.4   # Faster RCNN confidence threshold
# FASTER_RCNN_MODEL             = "fasterrcnn"
# MASK_RCNN_MODEL               = "maskrcnn"

# FASTER_RCNN_TEXT_GRAPH    = "./faster_rcnn_resnet50_coco_2018_01_28.pbtxt"
# FASTER_RCNN_MODEL_WEIGHTS = "./frozen_inference_graph.pb"
# FASTER_RCNN_MODEL_CLASSES = "./mscoco_labels.names"

# MASK_RCNN_MODEL_PATH = "./mask_rcnn_inception_v2_coco_2018_01_28/"
# MASK_RCNN_TEXT_GRAPH = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

# -------------------------------------------------

DEFAULT_MODEL           = 'ssdmn1'
DEFAULT_FRAME_WAIT      = (10)
DEFAULT_INTERVAL        = (0)
DEFAULT_NMS_THRESHOLD   = 0.4

# -------------------------------------------------

# Parse config file for configuration
def get_config_file_parameters():
    parser = argparse.ArgumentParser(description='CNN-based oject recogniser | cloudwise.co | 2019')
    parser.add_argument('--config', help='path to config file', type=str, required=True)

    args = parser.parse_args()

    # Validate the configh file path
    if not isfile(args.config):
        print("ERR: config file: ", args.config, " not found")
        sys.exit(1)

    # Read the configuration file
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config)

    in_path = config['DEFAULT']['InPath']
    # Validate input source : file or stream...
    if (in_path):
        if isfile(in_path) or urlparse(in_path).scheme:
            _capture = cv.VideoCapture(in_path)
        else:
            print("ERR: input video source: ", in_path, " not found")
            sys.exit(1)
    else:
        # ...or default to a local webcam stream
        _capture = cv.VideoCapture(0)

    _headless = config['DEFAULT'].getboolean('Headless', fallback=False)
    _blur = config['DEFAULT'].getboolean('Blur', fallback=False)
    _showlabels = config['DEFAULT'].getboolean('ShowLabels', fallback=True)
    _model = config.get('DEFAULT', 'Model', fallback=DEFAULT_MODEL)
    _noframewait = config['DEFAULT'].getint('NoFrameWait', fallback=DEFAULT_FRAME_WAIT)
    _interval = config['DEFAULT'].getint('Interval', fallback=DEFAULT_INTERVAL)
    _NMSthreshold = config['DEFAULT'].getfloat('NMSThreshold', fallback=DEFAULT_NMS_THRESHOLD)  # Currently fixed :-(

    detect_classes = config[_model]['Detect']
    # Split detect list : use list > set > list to remove any duplicate classes
    _detect = [int(item) for item in detect_classes.split(',')]

    _outpath = config['DEFAULT']['OutPath']
     # Validate the output file path...if set 
    if (_outpath):
        # Get the output path for images
        if not exists(_outpath):
            print("INFO: output path not found...creating:", _outpath)
            mkdir(_outpath)
            if not exists(_outpath):
                print("ERR: can't create output path: ", _outpath)
                sys.exit(1)

    _classes = config[_model]['ClassPath']
    _weights = config[_model]['WeightsPath']
    _graph = config[_model]['GraphPath']
    _threshold = config[_model].getfloat('Threshold')
     # Validate the classes, graph and weights file paths 
    if not isfile(_graph):
        print("ERR: model graph file: ", _graph, " not found")
        sys.exit(1)
    if not isfile(_weights):
        print("ERR: model weights file: ", _weights, " not found")
        sys.exit(1)
    if not isfile(_classes):
        print("ERR: classes file: ", _classes, " not found")
        sys.exit(1)    

    return _capture, _outpath, _headless, _showlabels, _threshold, \
        _detect, _blur, _model, _noframewait, _interval, _graph, _weights, _classes
