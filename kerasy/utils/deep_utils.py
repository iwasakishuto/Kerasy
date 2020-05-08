# coding: utf-8
import numpy as np
from .generic_utils import handleKeyError, handleTypeError

def set_weight(ver, val):
    if ver.shape != val.shape:
        raise TyoeError(f"weight shape must be the same. ({ver.shape} != {val.shape})")
    ver = val

def mk_class_get(all_classes={}, kerasy_abst_class=[], genre=""):
    def get(identifier):
        f"""
        Retrieves a Kerasy {genre.capitalize()} instance.
        @params identifier : {genre.capitalize()} identifier, string name of a {genre}, or
                             a Kerasy {genre.capitalize()} instance.
        @return {genre:<11}: A Kerasy {genre.capitalize()} instance.
        """
        if isinstance(identifier, str):
            handleKeyError(lst=list(all_classes.keys()), identifier=identifier)
            instance = all_classes.get(identifier)()
        else:
            handleTypeError(types=[str] + kerasy_abst_class, identifier=identifier)
            instance = identifier
        return instance
    return get

def get_params_size(layer):
    trainable_params = np.sum([layer.__dict__.get(key).size for key in layer._trainable_weights]).astype(int)
    not_trainable_params = np.sum([layer.__dict__.get(key).size for key in layer._non_trainable_weights]).astype(int)
    total_params = trainable_params+not_trainable_params
    return (total_params, trainable_params, not_trainable_params)

def print_summary(self):
    col_widths = (28,25,9)
    w1,w2,w3 = col_widths
    line_length = sum(col_widths)+len(col_widths)

    def arange_row_content(layer_type, output_shape, n_params):
        return f"{str(layer_type)[:w1]:<{w1}} {str(output_shape)[:w2]:<{w2}} {str(n_params)[:w3]:<{w3}} "

    print("-"*line_length)
    print(arange_row_content("Layer (type)", "Output Shape", "Param #"))
    print("="*line_length)

    tots = ables = nons = 0
    for i,layer in enumerate(self.layers):
        if i>0: print("-"*line_length)
        tot, able, non = get_params_size(layer)
        tots+=tot; ables+=able; nons+=non
        print(arange_row_content(
            f"{layer.name} ({layer.__class__.__name__})",
            str((None,) + layer.output_shape),
            tot,
        ))

    print("="*line_length)
    print(f"Total params: {tots:,}")
    print(f"Trainable params: {ables:,}")
    print(f"Non-trainable params: {nons:,}")
    print("-"*line_length)
