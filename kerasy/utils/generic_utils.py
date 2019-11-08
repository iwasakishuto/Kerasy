import sys
from collections import defaultdict

_UID_PREFIXES = defaultdict(int)

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
