### recog : a brief set of notes that hopefully help

##### Pre-requsiite : OpenCV

Please install 4.0.0 ideally or, if not available, v3.4.3+ as older versions will throw a CNN model exceptions given there is no cv.dnn functionality supported

```sh
$ pip install opencv-python
```

If you are running without the need or access to an X server, there is the (lighter-weight) headless python package which can be installed:

```sh
$ pip install opencv-python-headless
```

##### Pre-requisite : model weight files

Each model requires a weights file to go with the pbtext or cfg file that describes the model architecture. Alas these weights files are huge to the point of exceeding the GitHub file upload limit so are not included here.

So...please take a look at the top of recog_argparse.py for the names of the relevant files to Google for...and I'll add links to the necessary files for SSD Mobilenet v1 & v2 and YOLO v3 models here

Once you have the weights file, place it in the relevant 'model' sub-directory and use the path & name with the --weights argument as shown in the next section

##### Usage 

Run `python3 recog.py --help` for the full list of arguments which should be fairly self-explanatory.

The mandatory arguments are the `model` which defines which one should be used, `classes` which is the path to the text file contatining the list of objects the model was trained to recognise, `graph` which is the path to the text file defining the model architecture and `weights` which is the binary file containing the weights for the model. Note this latter file is not included and needs to be downloaded to your local `./models/...` directory (see above)

```sh
$ python3 recog.py --stream='http://192.168.0.1/video.cgi' --classes='./models/yolo3/yolo3.classes' --weights='./models/yolo3/yolo3.weights' --graph='./models/yolo3/yolo3.cfg' --model='yolo3'
$ python3 recog.py --video=people.mp4  --classes='<path>'  --weights='<path>'  --graph='<path>' --model='<ssd1, ssd2 or yolo3>'
$ python3 recog.py                     --classes='<path>'  --weights='<path>'  --graph='<path>' --model='<ssd1, ssd2 or yolo3>'
(no input source argument = use your local webcam as video source)
```

##### Sample video : licence-free content for testing and threshold-tuning

If you a sample video file of people to tune/test or simply experiment, see the link below. The Pexel site also has other
royalty-free

https://videos.pexels.com/videos/time-lapse-video-of-runners-855789


##### Headless OpenCV install on AWS Linux

The one-micro-instance-per-month is quite appealing and so I configure a micro EC2 instance to run recog. It requires some additional pre-requisites to be installed in order to build OpenCV 4 and, assuming you are running this without X as the output, one can also install the headless version of the OpenCV python package.

Remember to add the `--headless` argument otherwise it will error

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
The supervisor install is not mandatory but is is a great solution to running detached python-based processs given these may go zombie once you logout or close your ssh session into the EC2 instance.

See http://supervisord.org/introduction.html for more info on a very powerful package


##### Background reading

There are a ton of articles on Medium and elsewhere on the model architecture, performance (mAP accuracy vs inference speed), training and usage with OpenCV as we are doing here. But a couple of links to get started... 

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
