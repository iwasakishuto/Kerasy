from __future__ import absolute_import
from collections import defaultdict

_UID_PREFIXES = defaultdict(int)

def get_uid(prefix=""):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]
