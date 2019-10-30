import json
import numpy as np

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(npEncoder, self).default(obj)

class Params():
    __hidden_params__ = []
    __np_params__ = []
    __inf_params__ = []
    __initialize_method__ = ['list2np']

    def load_params(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)
        for method in self.__initialize_method__:
            self.__getattribute__(method).__call__()

    def save_params(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, cls=npEncoder)

    def list2np(self):
        for k,v in self.__dict__.items():
            if k in self.__np_params__:
                self.__dict__[k] = np.array(v)

    def replaceINF(self):
        for k,v in self.__dict__.items():
            if k in self.__inf_params__:
                v = np.array(v)
                self.__dict__[k] = np.where(v==np.inf, self.inf, v)

    def params(self):
        key_title='parameter'; val_title="value"
        keys = [key for key in self.__dict__.keys() if key not in self.__hidden_params__]
        vals = [self.__dict__[key] if self.__dict__[key] is not None else 'None' for key in keys]
        digit = max([len(val_title)] + [len(str(val)) for val in vals])
        width = max([len(key_title)] + [len(key) for key in keys])
        print(f"|{key_title:^{width}}|{val_title:^{digit}}|")
        print('-'*(width+digit+3))
        for i,(key,val) in enumerate(zip(keys,vals)):
            print(f"|{key:<{width}}|{val:>{digit}}|")
