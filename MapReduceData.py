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
            return MapReduceData(map(f, self.data))
    
    
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
        
    
    def filterKeys(self, keyList=[]):
        keyList = typeCast.listify(keyList)
        if len(keyList) == 0:
            return self
        
        if self.depth > 0:
            ret_data = dict((k, v.filterKeys(keyList))  for k, v in self.data.items())
            return MapReduceData(ret_data, self.depth)
        else:
            return MapReduceData(map(lambda x: dict((k, v) for k, v in x.items() if k in keyList),
                                     self.data))
    
    
    def getDataSet(self, keyList=[]):
        if len(keyList) != 0:
            return self.filterKeys(keyList).getDataSet()
        
        if self.depth > 0:
            ret = []
            for v in self.data.values():
                fn, ds = v.getDataSet()
                ret.append(ds)
            ret = np.vstack(ret)
            return fn, ret
        else:
            self.flatten()
            fn = [k for k in self.data[0]]
            ret = [list(sample.values()) for sample in self.data]
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
    
    def _flatten_subroutine(self):
        reloop=True
        while(reloop): #If a flatten happen, perform a recheck as more nested could have been uncovered
            reloop=False
            #Loop over all entries
            for d in self.data:
                to_add = []
                to_remove = []
                #Loop over the keys of an entry and check what to remove/add
                for k in d:
                    if type(d[k]) == type([]):
                        reloop=True
                        to_remove.append(k)
                        for idx, x in enumerate(d[k]):
                            to_add.append((k+'_{}'.format(idx)), x)
                #Add and remove keys (you can't do this while iterating over the dictionary)
                for k, v in to_add:
                    d[k] = v
                for k in to_remove:
                    d.pop(k)
                    
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        