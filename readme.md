#### A psuedo-random set of notes that might help

##### Useful references

https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
https://github.com/opencv/opencv_extra/tree/master/testdata/dnn

##### Installation on Linux: (Mask RCNN example)

`wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz`

`tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz`

##### Install on MacOS: (Mask RCNN example)

`curl -O http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz`

`tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz`

##### OpenCV installation:

Install 4.0.0 - using versions before 3.4.3 will throw CNN moddel exceptions given this functionality was not supported:

`pip install opencv-python`

If you are running without the need or access to an X server, there is the (lighter-weight) headless python package which can be installed:

`pip install opencv-python-headless`

##### TF graph creation:

The pbtxt file for the associated CNN model architecture should be on the tf or opencv GitHub - see Useful References above - but, if not, can be generated thus:

`python tf_text_graph_faster_rcnn.py --input /path/to/model.pb --config /path/to/example.config --output /path/to/graph.pbtxt`

Example:

`python tf_text_graph_ssd.py --input ./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --config ./ssd_mobilenet_v2_coco.config --output ./ssd_mobilenet_v2_coco_2019_01_28.pbtxt`

##### Usage examples:

`python3 recog.py --video=people.mpg`

`python3 recog.py --stream='http://192.168.0.1/video.cgi'`

`python3 recog.py`  
(uses your local webcam as video source)

##### Licence-free video clips for testing and paramter-tuning:

https://videos.pexels.com/videos/time-lapse-video-of-runners-855789


##### Headless OpenCV install on AWS Linux:

`sudo yum update`

`sudo yum install git cmake gcc-c++ cmake3`

`git clone https://github.com/Itseez/opencv.git`

`cd opencv`

`mkdir ./build`

`git checkout`

`cd ./build`

`cmake3 ../`

`sudo yum install numpy python-devel pip`

`sudo pip install opencv-python`

`pip install opencv-python-headless --user`

`pip install supervisor`
(see http://supervisord.org/introduction.html)
