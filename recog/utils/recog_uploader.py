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
from os.path import isfile, join, exists
import time
from configparser import ConfigParser, ExtendedInterpolation

# -------------------------------------------------

DEFAULT_INTERVAL    = (10)   # secs to sleep between scans
DEFAULT_LOCALPATH   = './out'
DEFAULT_REMOTEPATH  = './'

# -------------------------------------------------

# Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to run the recog FTP uploader')

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
    _delete = config['Upload'].getboolean('LocalDelete', fallback=False)
    _interval = config['Upload'].getint('UploadInterval', fallback=DEFAULT_INTERVAL)

    if (_localpath):
        if not exists(_localpath):
            print("ERR: local path:", _localpath, " not found")
            sys.exit(1)
   
    return _host, _user, _password, _localpath, _remotepath, _delete, _interval

# -------------------------------------------------

if __name__ == '__main__':
    # Extract the various command line parameters
    host, user, pwd, localpath, remotepath, delete, interval = get_arguments()
    
    try:
        ftp = ftplib.FTP(host, user, pwd)
        # ftp.login()
    except ftplib.error_reply as error:
        print("ERR: can't login to FTP server {} : {}".format(host, error))
        sys.exit(1)

    try:
        ftp.cwd(remotepath)
    except ftplib.error_reply as error:
        print("ERR: can't CWD to remote directory {} : {}".format(remotepath, error))
        ftp.quit()
        sys.exit(1)

    while True:
        # Get list of local filenames to upload
        print('INFO: scanning {}'.format(localpath))
        filenames = [f for f in listdir(localpath) if isfile(join(localpath, f))]

        if delete:
            print('INFO: uploading AND deleting...{}'.format(filenames))
        else:
            print('INFO: uploading...{}'.format(filenames))

        # Attempt to upload each file to the remote destination
        for filename in filenames:
            localpathfile = join(localpath, filename)
            try:
                with open(localpathfile, 'rb') as lf:
                    ftp.storbinary('STOR ' + filename, lf)
                    if delete:
                        # No exception so assume upload == successful...and delete local file
                        remove(localpathfile)
            except ftplib.error_perm as error:
                print('ERR: unable to upload file {} : {}'.format(filename, error))
                pass
        
        # Sleep
        time.sleep(interval)

    # FTP tidy-up
    ftp.quit()
    