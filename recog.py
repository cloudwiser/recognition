# -------------------------------------------------
# CNN-based Object Recogniser
# 
# With thanks to Satya Mallick & Sunita Nayak
# https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN
# https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
#
# Nick Hall : cloudwise.co
# 
# copyright cloudwise consulting 2019
# -------------------------------------------------

import sys
import cv2 as cv
import argparse
import numpy as np
import os.path
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

DEFAULT_THRESHOLD    = 0.75             # Confidence threshold - was 0.3
DEFAULT_OBJECT_CLASS = {1}              # Person from the COCO set   
SSD_MODEL            = "ssd"
YOLO_MODEL           = "yolo3"

# MODEL_PATH = "./mask_rcnn_inception_v2_coco_2018_01_28/"
# TEXT_GRAPH = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
SSD_MODEL_PATH = "./ssd_mobilenet_v2_coco_2018_03_29/"
SSD_TEXT_GRAPH = "./ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
SSD_MODEL_WEIGHTS = SSD_MODEL_PATH + "frozen_inference_graph.pb"

YOLO3_TEXT_GRAPH = "./yolov3.cfg"
YOLO3_MODEL_WEIGHTS = "./yolov3.weights"

DEFAULT_OUTPUT_PATH = './out'
WIN_NAME = 'recog : cloudwise.co : '

NO_FRAME_SLEEP      = (30 * 1)

CV_TEXT_SIZE        = 0.5
CV_BOUNDING_COLOR   = (255, 178, 50)

# -------------------------------------------------

# For each frame, draw a bounding box for each detected object
def detectTFObjectsInFrame(frame, classes, detect_classes, predictions, threshold, showlabels, blur):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    
    _found = 0
    # num_classes = masks.shape[1]
    num_detections = predictions.shape[2]
    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(num_detections):
        box = predictions[0, 0, i]
        score = box[2]

        if score > threshold:
            class_id = int(box[1])

            # Is this class one that we are interested in?
            if class_id not in detect_classes:
                continue
            else:
                _found += 1

            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])
            
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Draw bounding box on the image
            cv.rectangle(frame, (left, top), (right, bottom), CV_BOUNDING_COLOR, 1)

            # Blur the bounding box for privacy?
            if blur:
                blur_region = frame[top:bottom, left:right]
                # apply a gaussian blur on the bounding region
                blur_region = cv.GaussianBlur(blur_region, (23, 23), 30)
                # merge this blurry rectangle into the frame
                frame[top:top + blur_region.shape[0], left:left + blur_region.shape[1]] = blur_region                

            # Show the object info?
            if showlabels:
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
    return _found


def detectYOLO3ObjectsInFrame(frame, classes, detect_classes, predictions, threshold):
    _found = 0
    class_ids = []
    confidences = []
    boxes = []
    Width = frame.shape[1]
    Height = frame.shape[0]
    nms_threshold = 0.4

    for out in predictions:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:

                # Is this class one that we are interested in?
                if class_id not in detect_classes:
                    continue
                else:
                    _found += 1

                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv.rectangle(frame, (round(x), round(y)), (round(x+w), round(y+h)), (255, 178, 50), 2)
    return _found


# Parse arguments
def getArguments():
    parser = argparse.ArgumentParser(description='Use this script to run the object recogniser')
    parser.add_argument('--video', help='path to video file')
    parser.add_argument('--stream', help='path to video stream')
    parser.add_argument('--out', help='path to output directory')
    parser.add_argument('--headless', help='disable X-server output', action='store_true')
    parser.add_argument('--showlabels', help='enable object labels', action='store_true')
    parser.add_argument('--blur', help='blur object region(s)', action='store_true')
    parser.add_argument('--threshold', help='set the detection threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--classes', help='[comma-delimited] list of COCO or YOLO classes', type=str)
    parser.add_argument('--model', help='CNN model : set to yolo3 or ssd]', type=str)
    args = parser.parse_args()

    _outpath = None
    _headless = False
    _showlabels = False
    _blur = False
    _threshold = DEFAULT_THRESHOLD
    _detect_classes = DEFAULT_OBJECT_CLASS
    _model = SSD_MODEL

    # Process the command line arguments
    if (args.showlabels):
        _showlabels = True
    if (args.headless):
        _headless = True
    if (args.blur):
        _blur = True
    if (args.threshold):
        _threshold = float(args.threshold)
    if (args.classes):
        _detect_classes = [int(item) for item in args.classes.split(',')]
        # Use list > set > list to remove any duplicate classes
        _detect_classes = list(set(_detect_classes))
    if (args.model):
        _model = str.lower(args.model)
    if (args.out):
        # Get the output path for images
        if not os.path.exists(args.out):
            print("INFO: output path:", args.out, " doesn't exist...creating:", args.out)
            os.mkdir(args.out)
            if os.path.exists(args.out):
                _outpath = args.out
            else:
                print("ERR: can't create output path: ", args.out)
                sys.exit(1)
        else:
            _outpath = args.out
    # Parse the command line args for the capture source
    if (args.video):
        # Open a video file
        if not os.path.isfile(args.video):
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
    return _capture, _outpath, _headless, _showlabels, _threshold, _detect_classes, _blur, _model

# COCO classes file loader
def loadCOCOclasses(classes_file_path):
    # Load names of COCO classes
    _classes = None
    with open(classes_file_path, 'rt') as f:
        _classes = f.read().rstrip('\n').split('\n')
    return _classes

# YOLO classes file loader
def loadYOLOclasses(classes_file_path):
    classes = None
    with open(classes_file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# TF net loader
def loadTFnet(model_weights, text_graph):
    # Load the network
    _net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
    _net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    _net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return _net

# YOLOv3 net loader
def loadYOLO3net(weights, config):
    # Load the network
    net = cv.dnn.readNet(weights, config)
    return net

# YOLOv3 output layer retrieval
def getYOLO3output(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Get the TF candidate object boxes
def getTFobjects(region, net, net_params):
    # Create a 4D blob from the region
    # blob = cv.dnn.blobFromImage(cv.resize(region, (300,300)), 1.0, (300, 300), swapRB=True, crop=False)
    blob = cv.dnn.blobFromImage(region, swapRB=True, crop=False)
    net.setInput(blob)
    # Run the forward pass to get object boxes from the output layers
    return net.forward(net_params)

# Get the YOLO3 candidate object boxes
def getYOLO3objects(region, net, net_params):
    scale = 0.00392
    # Create a 4D blob from the region
    blob = cv.dnn.blobFromImage(region, scale, (416, 416), (0,0,0), True, crop=False)
    net.setInput(blob)
    # Run the forward pass to get object boxes from the output layers
    return net.forward(net_params)

# -------------------------------------------------

if __name__ == "__main__":
    # Extract the video source and output directory for annotated images
    capture, outpath, headless, showlabels, threshold, detect_classes, blur, model = getArguments()

    # Load the relevant classes file
    # classes = loadYOLOClasses("yolov3.classes")
    classes = loadCOCOclasses("mscoco_labels.names")

    # Load the graph and weights for the SSD mobilenet model
    net = loadTFnet(SSD_MODEL_WEIGHTS, SSD_TEXT_GRAPH)
    
    # Set the output window name (assuming there is a GUI output path)
    if not headless:
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)

    # Frame processing loop
    print("INFO: starting frame processing...")
    while True:
        # Get a frame from the video/image/stream
        hasFrame, frame = capture.read()
        
        # Skip and sleep if there is no frame
        if not hasFrame:
            print("WARN: no frame...sleep for {} sec(s)".format(NO_FRAME_SLEEP))
            time.sleep(NO_FRAME_SLEEP)
            continue

        # TF SSD mobilenet : get the object predictions
        predictions = getTFobjects(frame, net, None)

        # TF Mask RCNN : get the object & mask predictions
        # predictions, masks = getTFobjects(frame, net, ['detection_out_final', 'detection_masks'])

        # YOLO v3 : get the object predictions
        # predictions = getYOLO3objects(frame, net, getYOLO3OutputLayers(net))
        
        # TF SSD mobilenet : annotate frame with bounding box, label (and mask) for objects of interest that exceed threshold
        found = detectTFObjectsInFrame(frame, classes, detect_classes, 
                                        predictions, threshold, showlabels, blur)

        # Watermark the frame
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        t, _ = net.getPerfProfile()
        performance = ' : inference=%0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
        height, width = frame.shape[:2]
        modelused = ' : ' + model + '@' + str(width) + 'x' + str(height)
        label = WIN_NAME + timestamp + performance + modelused
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, CV_TEXT_SIZE, (0, 0, 0), 1)

        # Write the frame to output directory
        if found > 0 and outpath:
            outputFile = outpath + '/' + timestamp + '.jpg'
            cv.imwrite(outputFile, frame.astype(np.uint8))

        # Display the frame to X if there is a GUI path
        if not headless:
            cv.imshow(WIN_NAME, frame)
        
        # Esc to quit
        if not headless and cv.waitKey(1) == 27: 
            frame.release()
            break

print("INF: stopped frame processing")
if not headless:
    cv.destroyAllWindows()
