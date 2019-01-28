# -------------------------------------------------
#  Mask RCNN-based Person Detector
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

# For each frame, draw a bounding box for each detected object
def detectObjectsInFrame(frame, classes, detect_classes, boxes, masks, confidence_thresh):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    
    found = False
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
                found = True

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
            cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    return found

# Parse arguments for video source
def getVideoSource():
    parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN person detector')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--stream', help='Path to video stream')
    args = parser.parse_args()
    # Parse any command line args and setup the capture source and output file
    if (args.video):
        # Open a video file
        if not os.path.isfile(args.video):
            print("Input video file: ", args.video, " doesn't exist")
            sys.exit(1)
        capture = cv.VideoCapture(args.video)
    elif (args.stream):
        # Open a video stream
        if not urlparse(args.stream).scheme:
            print("Input video stream: ", args.stream, " doesn't exist")
            sys.exit(1)
        capture = cv.VideoCapture(args.stream)
    else:
        # ...or default to a local webcam stream
        capture = cv.VideoCapture(0)
    return capture

# COCO classes file loader
def loadCOCOclasses(classes_file_path):
    # Load names of COCO classes
    classes = None
    with open(classes_file_path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

# TF DNN loader loader
def loadTFDNN(modelWeights, textGraph):
    # Load the network
    net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net

# -------------------------------------------------

if __name__ == "__main__":
    # Initialize our parameters
    confidence_thresh = 0.5  # Confidence threshold - was 0.5
    detect_classes    = {0}  # only trigger on people in MSCOCO set   

    # Extract the video source
    capture = getVideoSource()
    
    # Load the COCO classes
    classes = loadCOCOclasses("mscoco_labels.names")

    # Load the graph and weights for the CNN model
    model_path = "./mask_rcnn_inception_v2_coco_2018_01_28/"
    textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    modelWeights = model_path + "frozen_inference_graph.pb"
    net = loadTFDNN(modelWeights, textGraph)
    
    # Set the output window name (assuming there is a GUI output path)
    winName = 'Mask-RCNN Person Detection'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    # Frame processing loop
    while cv.waitKey(1) < 0:
        # Get a frame from the video/image/stream
        hasFrame, frame = capture.read()
        
        # Bail if we've reached end of the input
        if not hasFrame:
            print("Processing complete!!!")
            cv.waitKey(3000)
            break

        # Create a 4D blob from the frame
        blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

        # Input the blob to the network
        net.setInput(blob)

        # Run the forward pass to get output from the output layers
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

        # Find the bounding box and mask for any selected and present objects
        found = detectObjectsInFrame(frame, classes, detect_classes, boxes, masks, confidence_thresh)

        # Output our inference performance at the top of the frame
        t, _ = net.getPerfProfile()
        label = 'Mask-RCNN inference time/frame : %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Write the frame with the detection boxes to disk
        if (found):
            outputFile = './out/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jpg'
            cv.imwrite(outputFile, frame.astype(np.uint8))

        # Display the output if there is a GUI output path
        cv.imshow(winName, frame)
