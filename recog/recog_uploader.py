# -------------------------------------------------
# recog_uploader : FTP uploader for (recog) binary output
# 
# Nick Hall
# 
# Copyright (c) 2019 cloudwise (http://cloudwise.co)
# -------------------------------------------------

import sys
import ftplib
import argparse
from os import listdir, remove
from os.path import isfile, join, exists, getsize, split
import time
from configparser import ConfigParser, ExtendedInterpolation
from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler

# -------------------------------------------------

DEFAULT_LOCALPATH   = './out'
DEFAULT_REMOTEPATH  = './'

# -------------------------------------------------

# Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to run the recog FTP uploader')
    parser.add_argument('--config', help='path to config.ini file', type=str, required=True)

    args = parser.parse_args()

    # Validate the configh file path
    if not isfile(args.config):
        print("ERR: config file: ", args.config, " not found")
        sys.exit(1)

    # Read the configuration file
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config)

    _host = config['Upload']['Host']
    _user = config['Upload']['Username']
    _password = config['Upload']['Password']
    _localpath = config.get('Upload', 'LocalPath', fallback=DEFAULT_LOCALPATH)
    _remotepath = config.get('Upload', 'RemotePath', fallback=DEFAULT_REMOTEPATH)
    _localdelete = config['Upload'].getboolean('LocalDelete', fallback=False)

    if (_localpath):
        if not exists(_localpath):
            print("ERR: local path:", _localpath, " not found")
            sys.exit(1)
   
    return _host, _user, _password, _localpath, _remotepath, _localdelete

# File creation watcher : see watchdog package for more info
class ImagesWatcher:
    def __init__(self, localpath):
        self.__localpath = localpath
        self.__event_handler = ImagesEventHandler()
        self.__event_observer = Observer()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__localpath,
            recursive=True
        )

# Fle creation event handler : see watchdog package for more info
class ImagesEventHandler(RegexMatchingEventHandler):
    IMAGES_REGEX = [r"\S+\.jpg"]

    def __init__(self):
        super().__init__(self.IMAGES_REGEX)

    def on_created(self, event):
        # Wait for the file size to stop increasing if it is still being written
        file_size = -1
        while file_size != getsize(event.src_path):
            file_size = getsize(event.src_path)
            time.sleep(1)
        # Respond to file_creation event
        self.process(event)

    def process(self, event):
        _pathfilename = event.src_path
        # print("DEBUG: file created: " + _pathfilename)
        # Split the filename from full path
        path, filename = split(_pathfilename)
        # Push file to remote path
        try:
            with open(_pathfilename, 'rb') as lf:
                ftp.storbinary('STOR ' + filename, lf)
                print('INFO: uploaded {}'.format(filename))
            if localdelete:
                # No exception so assume upload == successful...and delete local file
                remove(_pathfilename)
                print('INFO: deleted {}'.format(_pathfilename))
        except ftplib.error_perm as error:
            print('ERR: upload failed {} : {}'.format(filename, error))

# -------------------------------------------------

if __name__ == '__main__':
    # Extract the various command line parameters
    host, user, pwd, localpath, remotepath, localdelete = get_arguments()
    
    # print("DEBUG: h={} u={} p={} lp={} rp={} del={} int={}".format(host, user, pwd, localpath, remotepath, delete, interval))

    try:
        ftp = ftplib.FTP(host, user, pwd)
        # ftp.login()
    except ftplib.error_reply as error:
        print("ERR: FTP login to {} failed : {}".format(host, error))
        sys.exit(1)

    try:
        ftp.cwd(remotepath)
    except ftplib.error_reply as error:
        print("ERR: FTP CWD to {} failed : {}".format(remotepath, error))
        ftp.quit()
        sys.exit(1)

    ImagesWatcher(localpath).run()
    
    # FTP tidy-up
    ftp.quit()
    