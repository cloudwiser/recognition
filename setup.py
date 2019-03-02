from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 2 and sys.version_info.minor >= 2, \
    "The recognition repo is designed to work with Python 2.7 and greater." \
    + "Please install it before proceeding."

with open(join("recog", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='recog',
    py_modules=['recog'],
    version=__version__,# '0.1',
    install_requires=[
        'opencv>=4.0.0',
        'numpy'
        'watchdog'      # for recog_uploader.py
    ],
    description="CNN-based object recogniser",
    author="Nick Hall",
)