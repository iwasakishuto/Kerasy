# coding: utf-8
import os
import re
import json
import datetime
import numpy as np
from fractions import Fraction

from .generic_utils import handleKeyError
from .generic_utils import priColor
from . import UTILS_DIR_PATH

DICT_SORT_METHODS = ["rnd_is_last"]
DICT_SORT_FUNCS   = ["_dict_rnd_is_last"]

class KerasyJSONEncoder(json.JSONEncoder):
    """ Support the additional type for saving to JSON file. """
    def default(self, obj):
        #=== Numpy object ===
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        #=== Datetime object ===
        if isinstance(obj, datetime.datetime):
            return objisoformat()
        #=== Random State object ===
        if isinstance(obj, np.random.RandomState):
            dict_obj = dict(zip(
                ["MT19937", "unsigned_integer_keys", "pos", "has_gauss", "cached_gaussian"],
                obj.get_state()
            ))
            return dict_obj
        #=== Otherwise ===
        # Same as `super(KerasyJSONEncoder, self).default(obj)`
        return super().default(obj)

class Params():
    """ Each class that inherits from this class describes the parameters to display.
    ex)
    ```
    class Hoge(Params):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.disp_params = ["paramsA", "paramsB"]
            self.paramsA = 1
            self.paramsB = "1"
            self.paramsC = [1]
    ```
    > hoge = Hoge()
    > hoge.params()
    |Parameter|Value|
    -----------------
    |paramsA  |    1|
    |paramsB  |    1|
    """
    def __init__(self):
        self.disp_params = []

    def format_params(self, verbose=1, list_params=[], fraction_params=[], message=""):
        # Additional Method for arrangin parameters for suiting to the respective model.
        # If you want to add some other methods, please add like follows.
        message = self.fraction2float(fraction_params=fraction_params, message=message, retmessage=True)
        message = self.list2np(list_params=list_params, message=message, retmessage=True)
        message = self.setRandomState(message=message, retmessage=True)
        if verbose>0:
            print(message)

    def load_params(self, path=None, verbose=1, list_params=[], fraction_params=[], **kwargs):
        """Load parameters from json file.
        @params path            : JSON file path.
        @params list_params     : If some params want to remain list instance, please specify.
        @params fraction_params : If some params are writen as fraction, and want to be used as float, please specify.
        """
        list_params.append("disp_params")
        if path is None:
            path = os.path.join(UTILS_DIR_PATH, "default_params", f"{self.__class__.__name__}.json")
        message = f"Loading Parameters from {priColor.color(path, color='blue')}"
        with open(path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)
        self.format_params(verbose=verbose, list_params=list_params, fraction_params=fraction_params, message=message)

    def save_params(self, path, sort="rnd_is_last"):
        """ Saving parameters (=`self.__dict__`) """
        if sort not in DICT_SORT_METHODS:
            handleKeyError(DICT_SORT_METHODS, sort=sort)

        sort_func = dict(zip(
            DICT_SORT_METHODS,
            [self.__getattribute__(func_name) for func_name in DICT_SORT_FUNCS]
        ))[sort]

        new_dict = sort_func()
        with open(path, 'w') as f:
            json.dump(new_dict, f, indent=2, cls=KerasyJSONEncoder)

    def fraction2float(self, fraction_params=[], message="", retmessage=False):
        """ Convert Fraction to Float. """
        fraction_params = fraction_params if isinstance(fraction_params, list) else list(fraction_params)
        pattern = r"\d*\/\d"
        for k,v in self.__dict__.items():
            if isinstance(v, str) and re.search(pattern, v):
                self.__dict__[k] = float(Fraction(v))
                message += f"\nConverted {priColor.color(k, color='green')} from Fraction to Float."
            elif isinstance(v, list) and re.search(pattern, "".join(map(str,v))):
                self.__dict__[k] = [float(Fraction(e)) for e in v]
                message += f"\nConverted {priColor.color(k, color='green')} from Fraction to Float."
        if retmessage: return message

    def list2np(self, list_params=["disp_params"], message="", retmessage=False):
        """ Convert List to Numpy Array. """
        list_params = list_params if isinstance(list_params, list) else list(list_params)
        for k,v in self.__dict__.items():
            if isinstance(v, list) and k not in list_params:
                self.__dict__[k] = np.asarray(v)
                message += f"\nConverted {priColor.color(k, color='green')} type from list to np.ndarray."
        if retmessage: return message

    def setRandomState(self, message="", retmessage=False):
        for k,v in self.__dict__.items():
            if isinstance(v, dict) and len(v)==5 and "MT19937" in v:
                self.__dict__[k] = np.random.RandomState()
                self.__dict__[k].set_state(tuple(v.values()))
                message += f"\nSet {priColor.color(k, color='green')} the internal state of the generator."
        if retmessage: return message

    def params(self, key_title='Parameter', val_title="Value", max_width=65):
        """ Display All parameters (=`self.__dict__`) in tabular form. """
        if len(self.disp_params)==0:
            # Parameters without `self.disp_params`
            params_dict = dict([(k,v) for k,v in self.__dict__.items() if k!="disp_params"])
        else:
            # Only parameters in `self.disp_params`.
            params_dict = dict([(k,v) for k,v in self.__dict__.items() if k in self.disp_params])
        pnames = [k for k in params_dict.keys()]
        p_name_width = len(max(pnames + [key_title], key=len))
        val_width = len(max([str(v) for v in params_dict.values()] + [val_title], key=len))
        val_width = min(max_width-p_name_width-3, val_width)
        print(f"| {key_title:<{p_name_width}} | {val_title:>{val_width}}|")
        print('-'*(p_name_width+val_width+6))
        for i,(key,val) in enumerate(params_dict.items()):
            val = str(val).replace('\n', '\\n')
            print(f"| {key:<{p_name_width}} | {val[:val_width]:>{val_width}}|")

    def _dict_rnd_is_last(self):
        new_dict = dict()
        rnd_key = None
        for k,v in self.__dict__.items():
            if isinstance(v, np.random.RandomState):
                rnd_key,rnd_state = (k,v)
            else:
                new_dict[k] = v
        if rnd_key is not None:
            new_dict[rnd_key] = rnd_state
        return new_dict
