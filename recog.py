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
BOUNDING_COLOR       = (255, 178, 50)

# MODEL_PATH = "./mask_rcnn_inception_v2_coco_2018_01_28/"
# TEXT_GRAPH = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
MODEL_PATH = "./ssd_mobilenet_v2_coco_2018_03_29/"
TEXT_GRAPH = "./ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
MODEL_WEIGHTS = MODEL_PATH + "frozen_inference_graph.pb"
DEFAULT_OUTPUT_PATH = './out'
WIN_NAME = 'recog : cloudwise.co : '
NO_FRAME_SLEEP = (30 * 1)
CV_TEXT_SIZE = 0.5

# -------------------------------------------------

# For each frame, draw a bounding box for each detected object
def detectObjectsInFrame(frame, classes, detect_classes, boxes, threshold, showlabels, blur):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    
    _found = 0
    # num_classes = masks.shape[1]
    num_detections = boxes.shape[2]
    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(num_detections):
        box = boxes[0, 0, i]
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
            cv.rectangle(frame, (left, top), (right, bottom), BOUNDING_COLOR, 1)

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
                cv.rectangle(frame, (left, top), (right, bottom), BOUNDING_COLOR, cv.FILLED)
                cv.putText(frame, label, (left, labeltop), cv.FONT_HERSHEY_SIMPLEX, CV_TEXT_SIZE, (0,0,0), 1)
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
    parser.add_argument('--classes', help='[comma-delimited] list of COCO object classes', type=str)
    args = parser.parse_args()

    _outpath = None
    _headless = False
    _showlabels = False
    _blur = False
    _threshold = DEFAULT_THRESHOLD
    _detect_classes = DEFAULT_OBJECT_CLASS

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
    return _capture, _outpath, _headless, _showlabels, _threshold, _detect_classes, _blur

# COCO classes file loader
def loadCOCOclasses(classes_file_path):
    # Load names of COCO classes
    _classes = None
    with open(classes_file_path, 'rt') as f:
        _classes = f.read().rstrip('\n').split('\n')
    return _classes

# TF DNN loader loader
def loadTFDNN(model_weights, text_graph):
    # Load the network
    _net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
    _net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    _net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return _net

# -------------------------------------------------

if __name__ == "__main__":
    # Extract the video source and output directory for annotated images
    capture, outpath, headless, showlabels, threshold, detect_classes, blur = getArguments()

    # Load the COCO classes
    classes = loadCOCOclasses("mscoco_labels.names")

    # Load the graph and weights for the CNN model
    net = loadTFDNN(MODEL_WEIGHTS, TEXT_GRAPH)
    
    # Set the output window name (assuming there is a GUI output path)
    if not headless:
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)

    # Frame processing loop
    print("INFO: starting frame processing...")
    while True:
        # Get a frame from the video/image/stream
        hasFrame, frame = capture.read()
        
        # Skip if there is no frame
        if not hasFrame:
            print("WARN: no frame...wait {} sec(s)".format(NO_FRAME_SLEEP))
            time.sleep(NO_FRAME_SLEEP)
            continue

        # Create a 4D blob from the frame
        blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

        # Input the blob to the network
        net.setInput(blob)

        # Run the forward pass to get output from the output layers
        # boxes, masks = net.forward(['detection_out_final', 'detection_masks'])    # Mask RCNN
        boxes = net.forward()                                                       # SSD Mobilenet

        # Find the bounding box and mask for any selected and present objects
        # found = detectObjectsInFrame(frame, classes, detect_classes, 
        #                               boxes, masks, threshold, blur)      # Mask RCNN
        found = detectObjectsInFrame(frame, classes, detect_classes, 
                                        boxes, threshold, showlabels, blur) # SSD Mobilenet

        # Output our inference performance at the top of the frame
        t, _ = net.getPerfProfile()
        label = WIN_NAME + ' time/frame = %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, CV_TEXT_SIZE, (0, 0, 0), 1)

        # Write the frame with the detection boxes to disk
        if found > 0 and outpath:
            outputFile = outpath + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jpg'
            cv.imwrite(outputFile, frame.astype(np.uint8))

        # Display the output if there is a GUI output path
        if not headless:
            cv.imshow(WIN_NAME, frame)
        
        # Esc to quit
        if not headless and cv.waitKey(1) == 27: 
            frame.release()
            break

print("INF: stopped frame processing")
if not headless:
    cv.destroyAllWindows()
