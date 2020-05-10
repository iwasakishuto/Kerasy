# coding: utf-8
import os
import re
import sys
import time
import datetime
import numpy as np
from six.moves.urllib.request import urlretrieve

from collections import defaultdict

def flatten_dual(lst):
    return [e for sublist in lst for e in sublist]

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

def get_varname(var, scope_=globals()):
    for varname,val in scope_.items():
        if id(val)==id(var):
            return varname

def disp_var_globals(*varnames, head_=True, align_=True, scope_=globals()):
    """
    def func():
        a = "hoge"
        b = 123
        c = [1,"1"]
        disp_var_globals("a","b","c",scope=locals())

    func()
    >>> a: hoge
    >>> b: 123
    >>> c: [1, '1']
    """
    if head_: print(f"#=== VARIABLE INFO ===")
    digit = max([len(e) for e in varnames]) if align_ else 1
    for var in varnames:
        print(f"{var:<{digit}}: {scope_.get(var)}")

def disp_val_globals(*values, head_=True, align_=True, scope_=globals()):
    """
    def func():
        a = "hoge"
        b = 123
        c = [1,"1"]
        disp_val_globals(a,b,c,scope=locals())

    func()
    >>> a: hoge
    >>> b: 123
    >>> c: [1, '1']
    """
    if head_: print(f"#=== VARIABLE INFO ===")
    names = [get_varname(val, scope_=scope_) for val in values]
    digit = max([len(e) for e in names]) if align_ else 1
    for name,val in zip(names, values):
        print(f"{name:<{digit}}: {val}")

def get_uid(prefix=""):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]

class priColor:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[36m'
    WHITE     = '\033[37m'
    RETURN    = '\033[07m'    # 反転
    ACCENT    = '\033[01m'    # 強調
    FLASH     = '\033[05m'    # 点滅
    RED_FLASH = '\033[05;41m' # 赤背景+点滅
    END       = '\033[0m'

    @staticmethod
    def color(value, color=None):
        if color is None:
            return str(value)
        else:
            color = color.upper()
            handleKeyError(priColor.__dict__.keys(), color=color)
            return f"{priColor.__dict__[color.upper()]}{value}{priColor.END}"

def handleKeyError(lst, msg_="", **kwargs):
    k,v = kwargs.popitem()
    if v not in lst:
        lst = ', '.join([f"'{e}'" for e in lst])
        raise KeyError(f"Please chose the argment `{k}` from {lst}.\n\033[32m{msg_}\033[0m")

def handleTypeError(types, msg_="", **kwargs):
    type2str = lambda t: re.sub(r"<class '(.*?)'>", r"\033[34m\1\033[0m", str(t))
    k,v = kwargs.popitem()
    if not any([isinstance(v,t) for t in types]):
        str_true_types  = ', '.join([type2str(t) for t in types])
        srt_false_type = type2str(type(v))
        if len(types)==1:
            err_msg = f"must be {str_true_types}"
        else:
            err_msg = f"must be one of {str_true_types}"
        raise TypeError(f"`{k}` {err_msg}, not {srt_false_type}.\n\033[32m{msg_}\033[0m")

def urlDecorate(url, addDate=True):
    """ Decorate URL like Wget. (Add datetime information and coloring url to blue.) """
    now = datetime.datetime.now().strftime("--%Y-%m-%d %H:%M:%S--  ") if addDate else ""
    return now + priColor.color(url, color="BLUE")

def measure_complexity(func, *args, repetitions_=10, **kwargs):
    times=0
    metrics=[]
    if "random_state" in kwargs:
        base_seed = kwargs.get("random_state")
        for i in range(repetitions_):
            kwargs["random_state"] = base_seed+i
            s = time.time()
            ret = func(*args, **kwargs)
            times += time.time()-s
            metrics.append(ret)
    else:
        for _ in range(repetitions_):
            s = time.time()
            ret = func(*args, **kwargs)
            times += time.time()-s
            metrics.append(ret)
    if metrics[0] is None:
        return times/repetitions_
    else:
        return (times/repetitions_, metrics)

def has_not_attrs(obj, *names):
    return [name for name in names if not hasattr(obj, name)]

def has_all_attrs(obj, *names):
    return sum([1 for name in names if not hasattr(obj, name)])==0

def handleRandomState(seed):
    """ Turn `np.random.RandomState` """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    raise ValueError(f"Could not conver {seed} to numpy.random.RandomState instance.")

def fout_args(*args, sep="\t"):
    return sep.join([str(e) for e in args])+"\n"

f_aligns           = ["<", ">", "=", "^"]
f_signs            = ["+", "-", " ", ""]
f_grouping_options = ["_", ",", ""]
f_types            = ["b", "c", "d", "e", "E", "f", "F", "g", "G", "n", "o", "s", "x", "X", "%"]

def format_spec_create(width=0, align=">", sign="", zero_padding=False,
                       grouping_option="", fmt=""):
    """
    Create a function which returns a formatted text.
    ~~~~~
    * Source Code  : https://github.com/python/cpython/blob/3.8/Lib/string.py
    * Documentation:  https://docs.python.org/3/library/string.html#format-specification-mini-language

    format_spec = [[fill]align][sign][#][0][width][grouping_option][.precision][type]
    =========================
    @params align           : [[fill]align]
    @params sign            : [sign]
    @params zero_padding    : [0]
    @params width           : [width]
    @params grouping_option : [grouping_option]
    @params fmt             : [.precision][type]
    @return lambda          : <function __main__.<lambda>(fill)>
    """
    handleKeyError(lst=f_aligns, align=align)
    handleKeyError(lst=f_signs,  sign=sign)
    handleKeyError(lst=f_grouping_options, grouping_option=grouping_option)
    if len(fmt)>0:
        handleKeyError(lst=f_types, fmt=fmt[-1])
    zero = "0" if zero_padding else ""
    handleTypeError(types=[int], width=width)
    return lambda fill : f"{fill:{align}{sign}{zero}{width}{grouping_option}{fmt}}"

def print_func_create(width=0, align=">", sign="", zero_padding=False,
                      grouping_option="", fmt="", color="black",
                      left_side_bar="", right_side_bar="",
                      left_margin=0, right_margin=0, end="\n"):
    """
    Create a function which prints a formatted text.
    Please see also the function `format_spec_create`.
    ==============================
    @params color                : string color
    @params left(right)_side_bar : (str)
    @params left(right)_margin   : (int)
    @params end                  : string appended after the last value, default a newline.
    @return lambda               : <function __main__.<lambda>(fill)>
    """
    format_spec = format_spec_create(width, align=align, sign=sign,
                                     zero_padding=zero_padding,
                                     grouping_option=grouping_option, fmt=fmt)
    def print_func(fill):
        info  = f"{left_side_bar}{' '*left_margin}"
        info += priColor.color(format_spec(fill), color=color)
        info += f"{' '*right_margin}{right_side_bar}"
        print(info, end=end)
    return print_func

class Table():
    def __init__(self):
        self.cols = {}
        self.table_width = 1
        self.head = None

    def _disp_title(self):
        for colname, options in self.cols.items():
            if "print_values" not in options:
                continue
            print_func = options.get("print_title")
            print_func(colname)
        print("|")

    def _disp_border(self, table_width=None, mark="="):
        table_width = self.table_width if table_width is None else table_width
        print(mark*table_width)

    def _disp_values(self, head=None):
        head = self.head if head is None else head
        for i in range(head):
            for colname, options in self.cols.items():
                if "print_values" not in options:
                    continue
                print_func = options.get("print_values")
                values = options.get("values")
                print_func(values[i])
            print("|")

    def show(self, head=None, table_width=None, mark="="):
        self._disp_title()
        self._disp_border(table_width=table_width, mark=mark)
        self._disp_values(head=head)

    def set_cols(self, colname, values, width=None, align=">", sign="",
                 zero_padding=False, grouping_option="", fmt="", color="black",
                 left_margin=0, right_margin=0):
        title_width = len(str(colname))
        if width is None:
            format_spec = format_spec_create(
                width=0, align=align, sign=sign, zero_padding=zero_padding,
                grouping_option=grouping_option, fmt=fmt
            )
            width = len(max([format_spec(v) for v in values], key=len))
        width = max(width, title_width)
        self.table_width += width + left_margin + right_margin + 1

        print_values = print_func_create(
            width=width, align=align, sign=sign, zero_padding=zero_padding,
            grouping_option=grouping_option, fmt=fmt, color=color,
            left_side_bar="|", right_side_bar="", end="",
            left_margin=left_margin, right_margin=right_margin,
        )
        print_title = print_func_create(
            width=width, align="^", sign="", zero_padding=False,
            grouping_option="", fmt="", color="ACCENT",
            left_side_bar="|", right_side_bar="", end="",
            left_margin=left_margin, right_margin=right_margin,
        )
        self.cols.update({colname: dict(
            print_values=print_values, print_title=print_title, values=values
        )})
        if self.head is None:
            self.head = len(values)
