# -------------------------------------------------
# CNN-based Person Detector
# 
# With huge thanks to Satya Mallick & Sunita Nayak
# https://github.com/spmallick/learnopencv/tree/master/Mask-RCNN
# https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
#
# Nick Hall : cloudwise : 2019
# -------------------------------------------------

import sys
import cv2 as cv
import argparse
import numpy as np
import os.path
import datetime
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

CONFIDENCE_THRESHOLD = 0.75    # Confidence threshold - was 0.3
DETECT_CLASSES       = {1}     # only trigger on people in MSCOCO set   

# MODEL_PATH = "./mask_rcnn_inception_v2_coco_2018_01_28/"
# TEXT_GRAPH = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
MODEL_PATH = "./ssd_mobilenet_v2_coco_2018_03_29/"
TEXT_GRAPH = "./ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
MODEL_WEIGHTS = MODEL_PATH + "frozen_inference_graph.pb"
DEFAULT_OUTPUT_PATH = './out'
WIN_NAME = 'CNN Person Detect'

# -------------------------------------------------

# For each frame, draw a bounding box for each detected object
def detectObjectsInFrame(frame, classes, detect_classes, boxes, confidence_thresh):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    
    _found = False
    # num_classes = masks.shape[1]
    num_detections = boxes.shape[2]
    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(num_detections):
        box = boxes[0, 0, i]
        score = box[2]

        if score > confidence_thresh:
            class_id = int(box[1])

            # Is this class one that we are interested in?
            if class_id not in detect_classes:
                continue
            else:
                _found = True

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
            cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 1)
    return _found

# Parse arguments
def getArguments():
    parser = argparse.ArgumentParser(description='Use this script to run the CNN-based person detector')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--stream', help='Path to video stream')
    parser.add_argument('--out', help='Path to output directory')
    parser.add_argument('--headless', help='Flag to inhibit any X output', action='store_true', default=False)
    args = parser.parse_args()
    # Parse the command line args for the output path
    _outpath = DEFAULT_OUTPUT_PATH
    _headless = False
    if (args.headless):
        _headless = True
    if (args.out):
        # Get the output path for images
        if not os.path.exists(args.out):
            print("Output path:", args.out, " doesn't exist - creating:", args.out)
            os.mkdir(args.out)
            if os.path.exists(args.out):
                _outpath = args.out
            else:
                print("Error creating output path: ", args.out)
                sys.exit(1)
        else:
            _outpath = args.out
    # Parse the command line args for the capture source
    if (args.video):
        # Open a video file
        if not os.path.isfile(args.video):
            print("Input video file: ", args.video, " doesn't exist")
            sys.exit(1)
        else:
            _capture = cv.VideoCapture(args.video)
    elif (args.stream):
        # Open a video stream
        if not urlparse(args.stream).scheme:
            print("Input video stream: ", args.stream, " doesn't exist")
            sys.exit(1)
        else:
            _capture = cv.VideoCapture(args.stream)
    else:
        # ...or default to a local webcam stream
        _capture = cv.VideoCapture(0)
    return _capture, _outpath, _headless

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
    capture, outpath, headless = getArguments()

    # Load the COCO classes
    classes = loadCOCOclasses("mscoco_labels.names")

    # Load the graph and weights for the CNN model
    net = loadTFDNN(MODEL_WEIGHTS, TEXT_GRAPH)
    
    # Set the output window name (assuming there is a GUI output path)
    if not headless:
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)

    # Frame processing loop
    print("Starting frame processing...")
    while True:
        # Get a frame from the video/image/stream
        hasFrame, frame = capture.read()
        
        # Skip if there is no frame
        if not hasFrame:
            print("No frame found - processing skipped")
            continue

        # Create a 4D blob from the frame
        blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

        # Input the blob to the network
        net.setInput(blob)

        # Run the forward pass to get output from the output layers
        # boxes, masks = net.forward(['detection_out_final', 'detection_masks'])    # Mask RCNN
        boxes = net.forward()                                                       # SSD Mobilenet

        # Find the bounding box and mask for any selected and present objects
        # found = detectObjectsInFrame(frame, classes, detect_classes, boxes, masks, confidence_thresh) # Mask RCNN
        found = detectObjectsInFrame(frame, classes, DETECT_CLASSES, boxes, CONFIDENCE_THRESHOLD)    # SSD Mobilenet

        # Output our inference performance at the top of the frame
        t, _ = net.getPerfProfile()
        label = WIN_NAME + ' time/frame : %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Write the frame with the detection boxes to disk
        if (found):
            outputFile = outpath + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jpg'
            cv.imwrite(outputFile, frame.astype(np.uint8))

        # Display the output if there is a GUI output path
        if not headless:
            cv.imshow(WIN_NAME, frame)
        
         # Esc to quit
        if not headless and cv.waitKey(1) == 27: 
            frame.release()
            break

print("Stopping frame processing")
if not headless:
    cv.destroyAllWindows()
