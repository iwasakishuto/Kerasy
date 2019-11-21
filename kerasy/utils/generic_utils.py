import os
import sys
from six.moves.urllib.request import urlretrieve
from collections import defaultdict

_UID_PREFIXES = defaultdict(int)

def get_file(fname, origin, cache_subdir='datasets', file_hash=None):
    """Downloads a file from a URL if not already in the cache.
    ref: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/data_utils.py#L123

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.kerasy`, placed in the cache_subdir `datasets`,
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
    cache_dir = os.path.join(os.path.expanduser('~'), '.kerasy')
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.kerasy')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)
        error_msg = 'URL fetch failure on {} : {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    return fpath


def get_uid(prefix=""):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]

class Progbar():
    """Displays a progress bar.
    @param target  : (int) Total number of steps expected. None if unknown.
    @param width   : (int) Progress bar width on screen.
    @param verbose : (int) Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    @param interval: (float) Minimum visual progress update interval (in seconds).
    """
    def __init__(self, target, width=30, verbose=1, interval=5e-2):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval

    def update(self, current, values=None):
        """Update the progress bar.
        @param current: (int) Index of current step.
        @param values : (list)
        """
        pass
