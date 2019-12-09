#coding: utf-8
import os
import re
import json
import datetime
import numpy as np
from fractions import Fraction

UTILS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

class AdditionalTypeSupportJSONEncoder(json.JSONEncoder):
    """ Support the additional type for saving to JSON file. """
    def default(self, obj):
        #=== Numpy object ===
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        #=== Datetime object ===
        elif isinstance(obj, datetime.datetime):
            return objisoformat()
        else:
            return super(npEncoder, self).default(obj)

class Params():
    def __init__(self):
        self.disp_params = []

    def format_params(self, verbose=1, list_params=[], fraction_params=[], message=""):
        # Additional Method for arrangin parameters for suiting to the respective model.
        message = self.fraction2float(fraction_params=fraction_params, message=message, retmessage=True)
        message = self.list2np(list_params=list_params, message=message, retmessage=True)
        if verbose>0:
            print(message)

    def load_params(self, path=None, verbose=1, list_params=["disp_params"], fraction_params=[], **kwargs):
        """Load parameters from json file.
        @params path      : JSON file path.
        @params verbose   : Display message if verbose > 0
        @params listparams: If some params want to remain list instance, please specify.
        """
        if path is None:
            path = os.path.join(UTILS_DIR_PATH, "default_params", f"{self.__class__.__name__}.json")
        message = f"Loading Parameters from '{path}'..."
        with open(path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)
        self.format_params(verbose=verbose, list_params=list_params, fraction_params=fraction_params, message=message)

    def save_params(self, path):
        """ Saving parameters (=`self.__dict__`) """
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, cls=SupportEncoder)

    def fraction2float(self, fraction_params=[], message="", retmessage=False):
        """ Convert Fraction to Float. """
        fraction_params = fraction_params if isinstance(fraction_params, list) else list(fraction_params)
        pattern = r"\d*\/\d"
        for k,v in self.__dict__.items():
            if isinstance(v, str) and re.search(pattern, v):
                self.__dict__[k] = float(Fraction(v))
                message += f"\nConverted {k} from Fraction to Float."
            elif isinstance(v, list) and re.search(pattern, "".join(map(str,v))):
                self.__dict__[k] = [float(Fraction(e)) for e in v]
                message += f"\nConverted {k} from Fraction to Float."
        if retmessage: return message

    def list2np(self, list_params=["disp_params"], message="", retmessage=False):
        """ Convert List to Numpy Array. """
        list_params = list_params if isinstance(list_params, list) else list(list_params)
        for k,v in self.__dict__.items():
            if isinstance(v, list) and k not in list_params:
                self.__dict__[k] = np.asarray(v)
                message += f"\nConverted {k} type from list to np.ndarray."
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
        print(f"|{key_title:<{p_name_width}}|{val_title:>{val_width}}|")
        print('-'*(p_name_width+val_width+3))
        for i,(key,val) in enumerate(params_dict.items()):
            val = str(val).replace('\n', '\\n')
            print(f"|{key:<{p_name_width}}|{val[:val_width]:>{val_width}}|")
