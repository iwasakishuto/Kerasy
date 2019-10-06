import json

class Params():
    __remove_params__ = []

    def load_params(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save_params(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def params(self):
        key_title='parameter'; val_title="value"
        keys = [key for key in self.__dict__.keys() if key not in self.__remove_params__]
        vals = [self.__dict__[key] if self.__dict__[key] is not None else 'None' for key in keys]
        digit = max([len(val_title)] + [len(str(val)) for val in vals])
        width = max([len(key_title)] + [len(key) for key in keys])
        print(f"|{key_title:^{width}}|{val_title:^{digit}}|")
        print('-'*(width+digit+3))
        for i,(key,val) in enumerate(zip(keys,vals)):
            print(f"|{key:<{width}}|{val:>{digit}}|")
