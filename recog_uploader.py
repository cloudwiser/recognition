# -------------------------------------------------
# recog FTP uploader
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
import datetime
import time

# -------------------------------------------------

SLEEP_INTERVAL = (30)   # secs to sleep between scans

# -------------------------------------------------

# Parse arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Use this script to run the recog FTP uploader')
    parser.add_argument('--host', help='name of FTP host')
    parser.add_argument('--username', help='FTP username')
    parser.add_argument('--password', help='FTP password')
    parser.add_argument('--out', help='path to local (recog output) directory')
    parser.add_argument('--remote', help='path to remote upload directory')
    parser.add_argument('--delete', help='delete local files on upload', action='store_true')
    args = parser.parse_args()

    _host = None
    _user = None
    _password = None
    _outpath = None
    _remotepath = None
    _delete = False

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

    if (args.password):
        _password = args.password
    return _host, _user, _password, _outpath, _remotepath, _delete

# -------------------------------------------------

if __name__ == '__main__':
    # Extract the various command line parameters
    host, username, password, outpath, remotepath, delete = get_arguments()

    try:
        ftp = ftplib.FTP(host, username, password)
        ftp.login()
    except ftplib.error_reply as error:
        print("ERR: can't login to FTP server: {}".format(error))
        sys.exit(1)

    try:
        ftp.cwd(remotepath)
    except ftplib.error_reply as error:
        print("ERR: can't CWD to remote directory: {}".format(error))
        ftp.quit()
        sys.exit(1)

    while False:
        # Get list of local filenames to upload
        filenames = [f for f in listdir(outpath) if isfile(join(outpath, f))]
        print('DEBUG: files to upload: {}'.format(filenames))

        # Attempt to upload each file to the remote destination
        for filename in filenames:
            localpathfile = join(outpath, filename)
            try:
                with open(localpathfile, 'rb') as lf:
                    ftp.retrbinary('STOR ' + filename, lf.write)
                    if delete:
                        # No exception so assume upload == successful...and delete local file
                        print('DEBUG: file to delete: {}'.format(localpathfile))
                        # remove(localpathfile)
            except ftplib.error_perm as error:
                print('ERR: unable to upload file {} : {}'.format(filename, error))
                pass
        # Sleep
        time.sleep(SLEEP_INTERVAL)

    ftp.quit()
