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
import numpy as np
import datetime
import time

from recog_helpers import *
from recog_dnn import *

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

SSD_MN1_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

APP_NAME = 'recog : cloudwise.co : '

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
    elif model == SSD_MN1_MODEL:
        classes = SSD_MN1_CLASSES
        net = load_Caffe_net(SSD_MN1_MODEL_WEIGHTS, SSD_MN1_TEXT_GRAPH)
    else:
        classes = load_COCO_classes(SSD_MN2_MODEL_PATH + "mscoco_labels.names")
        net = load_TF_net(SSD_MN2_MODEL_WEIGHTS, SSD_MN2_TEXT_GRAPH)

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
        elif model == SSD_MN1_MODEL:
            predictions = get_SSD_MobileNet1_objects(frame, net, None)
            found = objects_from_single_layer_output(frame, classes, detect_classes, predictions, threshold, showlabels, blur)
        else:   # model == SSD_MN2_MODEL:
            predictions = get_SSD_MobileNet2_objects(frame, net, None)
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
