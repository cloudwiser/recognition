See https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

Linux: 
(Mask RCNN)
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

MacOS: 
(Mask RCNN)
curl -O http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

OpenCV:
Install 4.0.0 if possible as using versions before 3.4.3 will throw CNN moddel exceptions:
pip install opencv-python

Graph creation:
e.g. python tf_text_graph_faster_rcnn.py --input /path/to/model.pb --config /path/to/example.config --output /path/to/graph.pbtxt

python tf_text_graph_ssd.py --input ./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --config ./ssd_mobilenet_v2_coco.config --output ./ssd_mobilenet_v2_coco_2019_01_28.pbtxt

Usage examples:
python3 person_detect.py --video=people.mpg
python3 person_detect.py --stream='http://192.168.0.1/video.cgi'
python3 person_detect.py  (uses local webcam as video source)

Test clips:
https://videos.pexels.com/videos/time-lapse-video-of-runners-855789
