# General class wrapper to perform map-reduce operation on text data loaded from an xml.
# The 'parserFunction' argument in loadXML is the function used to parse the xml tree.
# It is expected to take an lxml.etree.iterator and return a list of dicts
import typeCast

import numpy as np
from functools import reduce

class MapReduceData:
    def __init__(self, data, depth=0):
        assert depth >= 0
        self.data = data
        self.depth = depth
        return
    
    
    def map(self, f):
        if self.depth > 0:
            ret_data = dict((k, v.map(f)) for k, v in self.data.items())
            return MapReduceData(ret_data, self.depth)
        else:
            return MapReduceData(list(map(f, self.data)))
    
    
    def reduce(self, f, prefix=[]):
        prefix = typeCast.listify(prefix)
        if self.depth > 0:
            ret_data = dict((k, v.reduce(f, prefix)) for k, v in self.data.items())
            return MapReduceData(ret_data, self.depth-1)
        else:
            return reduce(f, prefix + self.data)
        
        
    def filter(self, f):
        if self.depth > 0:
            ret_data = dict((k, v.filter(f)) for k, v in self.data.items())
            return MapReduceData(ret_data, self.depth)
        else:
            return MapReduceData(filter(f, self.data))
    
    
    def groupByKey(self, key):
        if self.depth > 0:
            ret_data = dict((k, v.groupByKey(key)) for k, v in self.data.items())
            return MapReduceData(ret_data, self.depth+1)
        else:
            #Get all possible value and create a dictionayr to store them
            values = set(map(lambda x:x[key], self.data))
            di = dict((val, []) for val in values)
            #Fill the dictionary
            for d in self.data:
                di[d[key]].append(d)
            #Convert to MapReduceData and create parent
            for k in di:
                di[k] = MapReduceData(di[k])
            return MapReduceData(di, 1)
        
    
    def selectKeys(self, keyList=[]):
        keyList = typeCast.listify(keyList)
        if len(keyList) == 0:
            return self
        
        if self.depth > 0:
            ret_data = dict((k, v.filterKeys(keyList))  for k, v in self.data.items())
            return MapReduceData(ret_data, self.depth)
        else:
            return MapReduceData(list(map(lambda x: dict((k, v) for k, v in x.items() if k in keyList),
                                     self.data)))
    
    
    def getDataSet(self, keyList=[]):
        if len(keyList) != 0:
            return self.selectKeys(keyList).getDataSet()
        
        if self.depth > 0:
            ret = []
            for v in self.data.values():
                fn, ds = v.getDataSet()
                ret.append(ds)
            ret = np.vstack(ret)
            return fn, ret
        else:
            fn = [k for k in self.data[0]]
            #Reduce dark magic
            # _flatten_reduce() concatenate two element of a list into a np.array 
            # (with a lot of sanity check and is robust to concatenating list element with an array)
            ret = [reduce(self._flatten_reduce, [np.array([])] + list(sample.values())) for sample in self.data]
            #ret = [list(sample.values()) for sample in self.data]
            ret = np.array(ret)
            return fn, ret
    
    
    #Only work with nested list, other nest container (e.g. dict, set) won't be affected
    def flatten(self):
        if self.depth > 1:
            for k in self.data:
                self.data[k].flatten()
            return
        else:
            self._flatten_subroutine()
            return
    
    def _flatten_reduce(self, x, y):
        return np.concatenate((x,
                               y if type(y)==type(np.array([]))
                               else np.array(typeCast.listify(y))
                               ))
                    
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        