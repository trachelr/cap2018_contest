#Type cast and utilities
import json

import numpy as np

#Update a dict/set and return it
def update_return(a, b):
    a.update(b)
    return a


#All-time favorite listify
def listify(x):
    if type(x) == type([]):
        return x
    else:
        return [x]
    

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    