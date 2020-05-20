# coding: utf-8
import os
from six.moves.urllib import request
from six.moves.urllib.error import HTTPError, URLError

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.kerasy') # /Users/<username>/.kerasy
DATADIR_BASE = os.path.expanduser(CACHE_DIR)                 # /Users/<username>/.kerasy
# Check whether uid/gid has the write access to DATADIR_BASE
if not os.access(DATADIR_BASE, os.W_OK):
    DATADIR_BASE = os.path.join('/tmp', '.kerasy')

def get_file(fname, origin, cache_subdir='datasets', file_hash=None):
    """Downloads a file from a URL if not already in the cache.
    ref: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/data_utils.py#L123

    By default the file at the url `origin` is downloaded to the
    CACHE_DIR `~/.kerasy`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.kerasy/datasets/example.txt`.

    You have to make a directory `~/.kerasy` and `~./kerasy/datasets`,
    and check whether you can access these directories using `os.access("DIRECOTRY", os.W_OK)`
    If this method returns False, you have to change the ownership of them like
    ```
    $ sudo chown iwasakioshuto: ~/.kerasy
    $ sudo chown iwasakioshuto: ~/.kerasy/datasets
    ```
    """
    # /Users/<username>/.kerasy/`cache_subdir`
    DATADIR = os.path.join(DATADIR_BASE, cache_subdir)
    if not os.path.exists(DATADIR):
        os.makedirs(DATADIR)
    fpath = os.path.join(DATADIR, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)
        error_msg = 'URL fetch failure on {} : {} -- {}'
        try:
            try:
                request.urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    return fpath
