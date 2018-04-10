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
        if self.depth > 1:
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
    
    
    def getDataSet(self, keyList=[]):
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        