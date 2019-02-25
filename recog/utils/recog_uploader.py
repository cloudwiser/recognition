# -------------------------------------------------
# recog_uploader : FTP uploader for (recog) binary output
# 
# Nick Hall : cloudwise.co
# 
# copyright cloudwise consulting 2019
# -------------------------------------------------

import sys
import ftplib
import argparse
from os import listdir, remove
from os.path import isfile, join, exists
import time

# -------------------------------------------------

DEFAULT_SLEEP_INTERVAL = (10)   # secs to sleep between scans
DEFAULT_OUTPATH = './out'

# -------------------------------------------------

# Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to run the recog FTP uploader')
    parser.add_argument('--host', help='name of FTP host', required=True)
    parser.add_argument('--user', help='FTP username', required=True)
    parser.add_argument('--pwd', help='FTP password', required=True)
    parser.add_argument('--out', help='path to local (recog output) directory')
    parser.add_argument('--remote', help='path to remote upload directory', required=True)
    parser.add_argument('--delete', help='delete local files on upload', action='store_true')
    parser.add_argument('--interval', help='poll interval (secs)', type=int)
    args = parser.parse_args()

    _host = None
    _user = None
    _password = None
    _outpath = DEFAULT_OUTPATH
    _remotepath = None
    _delete = False
    _interval = DEFAULT_SLEEP_INTERVAL

    if (args.delete):
        _delete = True

    if (args.out):
        # Get the local path - defined by "recog.py --out=<path>""
        if not exists(args.out):
            print("ERR: output path:", args.out, " doesn't exist...exiting:", args.out)
            sys.exit(1)
        else:
            _outpath = args.out

    if (args.remote):
        # Get the remote path used for uploading
        _remotepath = args.remote

    if (args.host):
        _host = args.host

    if (args.user):
        _user = args.user

    if (args.pwd):
        _pwd = args.pwd

    if (args.interval):
        _interval = int(args.interval)
    return _host, _user, _pwd, _outpath, _remotepath, _delete, _interval

# -------------------------------------------------

if __name__ == '__main__':
    # Extract the various command line parameters
    host, user, pwd, outpath, remotepath, delete, interval = get_arguments()
    
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
        print('INFO: scanning {}'.format(outpath))
        filenames = [f for f in listdir(outpath) if isfile(join(outpath, f))]

        if delete:
            print('INFO: uploading AND deleting...{}'.format(filenames))
        else:
            print('INFO: uploading...{}'.format(filenames))

        # Attempt to upload each file to the remote destination
        for filename in filenames:
            localpathfile = join(outpath, filename)
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
    