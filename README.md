#### recog : a quick intro

##### Pre-requsiite : OpenCV

Please install/build OpenCV 4 or later given versions prior to v3.4.3 will throw a CNN model exception given the lack of `cv.dnn` support.

```sh
$ pip install opencv-python
```

If your environment is minus any UI output, there is the (lighter-weight) headless python package which can be installed:

```sh
$ pip install opencv-python-headless
```

##### Pre-requisite : model weight files

Each model requires a weights file to go with the pbtext or cfg file that describes the model architecture. 

The SSD MobileNet v1 file is in the repo but the SSD v2 and YOLO v3 weights files are huge to the point of exceeding the GitHub file upload limit and are not included.

If you wish to use these models, please review the models sections of `config.ini` for the relevant filename(s) to Google...although I'll add links to the necessary files here at some point.

Once you have the weights file, place it in the relevant 'model' sub-directory, configure the `WeightsPath` parameter in the relevant `[<model>]` section and, if necessary, also set the `Model` paramter in the `[DEFAULT]` section.

##### Usage 

`$ python3 recog.py --config=<path_to_config_file>`

See the example `config.ini` file in the repo for info on the settable parameters and what they are used for.


##### Sample video : licence-free content for testing and threshold-tuning

If you require a sample video file of people objects aka persons to tune or test or simply experiment with, see the link below. 
The Pexel site also has other royalty-free content that may be relevant.

https://videos.pexels.com/videos/time-lapse-video-of-runners-855789


##### Headless OpenCV install on AWS Linux

The one-free-micro-instance-per-month on AWS is a nice bargain and I have configured a micro EC2 instance to run 2 instances of recog handling 2 separate RTSP video streams. Inference processing time will depend on model, stream resolution and number of recog instances obviously.

It requires some additional pre-requisites to be installed in order to first build OpenCV 4 as below and, assuming you are running this without X as the output, you can also install the headless version of the OpenCV python package.

Remember to set the `Headless` paramter in the `[DEFAULT]` section to `true` in the `config.ini` file otherwise the script will error.

```sh
$ sudo yum update
$ sudo yum install git cmake gcc-c++ cmake3
$ git clone https://github.com/Itseez/opencv.git
$ cd opencv
$ mkdir ./build
$ git checkout
$ cd ./build
$ cmake3 ../
$ sudo yum install numpy python-devel pip
$ sudo pip install opencv-python
$ pip install opencv-python-headless --user
$ pip install supervisor
```
The `supervisor` install is not mandatory but is is an excellent solution for running detached python-based processs on Linux.
This is key if you are wanting to run continously and unattended given these processes can go zombie once you logout or close your ssh session into the EC2 instance.

See http://supervisord.org/introduction.html for more info on what is an amazingly feature-rich package.


##### Image upload via recog_uploader

This is a simple FTP uploader script that monitors a local directory and copies or moves local image files to a remote directory hosted on a FTP server.

See the example `config.ini` file in the repo for info on the settable parameters and what they are used for.

At the time of writing, the script doesn't support anonymous FTP login. Another TODO or...


##### Pull requests

These are very welcome :-)


##### Background reading

Given the continual research and application of CNN-based models to vision processing, there are a ton of articles on Medium and elsewhere that details the model architecture, performance (mAP accuracy vs inference speed), training as well as (for some) usage with OpenCV as we are doing here. A couple of relevant on the OpenCV aspects that might help if you are wanting to dive deeper... 

https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
https://github.com/opencv/opencv_extra/tree/master/testdata/dnn


##### TensorFlow graph file creation

You shouldn't need to recreate the pbtxt (graph) file but future changes in either Tensorflow and/or OpenCV dnn might necessitate it. Equally, if you decide to deploy a modified graph following re-training, the command line from the TF GH model repo is shown below...which I have not tested:

```sh
$ python tf_text_graph_faster_rcnn.py --input /path/to/model.pb --config /path/to/example.config --output /path/to/graph.pbtxt
```

Example:

```sh
$ python tf_text_graph_ssd.py --input ./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --config ./ssd_mobilenet_v2_coco.config --output ./ssd_mobilenet_v2_coco_2019_01_28.pbtxt
```
