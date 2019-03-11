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


##### OpenCV 4 install on Raspberry Pi Zero W

A word of warning on this : although recog works on an Pi Zero W, inference is really slow e.g. for the SSD v1 model, it was taking 250 secs for person detecting in a 1920 x 1080 frame

But, if you are still keen, download and flash the official Raspbian Stretch Lite image onto a 32GB SDD
Note: I can guarantee that an 8GB SDD isn't large enough and the `make` will then fail at the end of ~15 hours!

In the SSD `/root` directory, enable SSH via `touch ssh` and add a `wpa_supplicant.conf` file with your WiFi credentials
Eject it, insert into the Zero W and check you can `ssh` access over WiFi
Then expand the filesystem using `raspi-config` and do a `sudo reboot now`

The install steps below are based on https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/
BUT the `make` can take a very long time on a RPi Zero...so you may want to look at a cross-compile solution

Finally, having gone through the build steps below, don't forget to back-up the final SSD image in case it gets corrupted!

```sh
$ sudo apt-get upgrade
$ sudo apt-get update
$ sudo apt-get install python3-pip
$ sudo apt-get install cmake

$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev

$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libcanberra-gtk*
$ sudo apt-get install libatlas-base-dev gfortran

$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.0.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.0.0.zip
$ unzip opencv.zip
$ unzip opencv_contrib.zip
$ mv opencv-4.0.0 opencv
$ mv opencv_contrib-4.0.0 opencv_contrib
# Omit the virtual environment-related installs given this will be single environment

$ cd ~/opencv
$ mkdir build
$ cd build

# Neither NEON nor VFPv3 is supported on arm6 i.e. Zero W ...but try enabling VFPv2 (assuming it's a valid build flag)
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=OFF \
    -D ENABLE_VFPV2=ON \
    -D BUILD_TESTS=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF ..

$ sudo nano /etc/dphys-swapfile
    # Increase the swap size otherwise 'make' will fail 
    # CONF_SWAPSIZE=100
    CONF_SWAPSIZE=2048

$ sudo /etc/init.d/dphys-swapfile stop
$ sudo /etc/init.d/dphys-swapfile start

# Note: 'make j4' aka multi-core is not supported on the Zero W so this will be slow
# ...allow ~18 hours!
$ make

$ sudo make install
$ sudo ldconfig

$ sudo nano /etc/dphys-swapfile
    # Change the swap size back to the default setting
    CONF_SWAPSIZE=100
    # CONF_SWAPSIZE=2048

$ sudo /etc/init.d/dphys-swapfile stop
$ sudo /etc/init.d/dphys-swapfile start

$ cd ~/.local/lib/python3.5/site-packages/
$ ln -s /usr/local/python/cv2/python-3.5/cv2.cpython-35m-arm-linux-gnueabihf.so cv2.so
$ cd ~

# supervisor install for python 3+ 
# ...at the time of writing a production release of python 3 supervisor has still to be made to PyPi
$ sudo apt-get install python-pip git
$ pip install git+https://github.com/Supervisor/supervisor@master
```

##### OpenCV 4 install on AWS Linux

The one-free-micro-instance-per-month on AWS is a nice bargain and I have configured a micro EC2 instance to run 2 instances of recog handling 2 separate RTSP video streams. Inference processing time will depend on model, stream resolution and number of recog instances obviously.

It requires some additional pre-requisites to be installed in order to first build OpenCV 4 as below for python 2.7 and, assuming you are running this without X as the output, install the headless version of the OpenCV python package.

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

# supervisor install for python 2.7
$ pip install supervisor
```

The `supervisor` install is not mandatory but is is an excellent solution for running detached python-based processs on Linux.
This is key if you are wanting to run continously and unattended given these processes can go zombie once you logout or close your ssh session into the EC2 instance.

See http://supervisord.org/introduction.html for more info on an very feature-rich package.


##### Image upload via recog_uploader

This is a simple FTP uploader script that monitors a local directory and copies or moves local image files to a remote directory hosted on a FTP server.

`$ python3 recog_uploader.py --config=<path_to_config_file>`

See the example `[Upload]` section in the `config.ini` file in the repo for info on the settable parameters and what they are used for.

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
