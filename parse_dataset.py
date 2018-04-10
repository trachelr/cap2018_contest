# General class wrapper to perform map-reduce operation on text data loaded from an xml.
# The 'parserFunction' argument in loadXML is the function used to parse the xml tree.
# It is expected to take an lxml.etree.iterator and return a list of dicts
import datetime
import typeCast

import numpy as np

from lxml import etree
from functools import reduce

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence

#Default function for parsing Cambridge data
def defaultParserFunction_Cambridge(path):
    #Only get to the writings tag and process childs from here
    for event, elem in etree.iterparse(path):
        if elem.tag=='writings':
            break 
    print('Writings tag found with {} children(s)'.format(len(elem.getchildren())))
    
    ##First Pass
    print('Starting first pass')
    skipped = 0
    nbEntries = 0
    data = []
    report_freq = int(0.1 * len(elem.getchildren()))
    for idEntry, entry in enumerate(elem.getchildren()):
        if entry.tag != 'writing':
            print('Not a writing a entry ({} found). Skip'.format(entry.tag))
            skipped+=1
            continue
        #All writing entry should follow the same hierarchy
        local = {}
        attr = dict(entry.items())
        local['entry_id'] = attr['id']
        local['level'] = attr['level']
        local['unit'] = attr['unit']
        
        #Process child entries (text, learner, topic, date and grade)
        if len(entry.getchildren()) != 5:
            print('Entry with invalid number of children (expected 5, found {}). skipped'\
                  .format(len(entry.getchildren())))
            skipped+=1
            continue
        for child in entry.getchildren():
            attr = dict(child.items())
            
            #Parse Learner tag
            if child.tag == 'learner':
                local['learner_id'] = int(attr['id'])
                local['nationality'] = attr['nationality']
            
            #Parse topic tag
            elif child.tag == 'topic':
                local['topic_id'] = int(attr['id'])
            
            #Parse date tag
            elif child.tag == 'date':
                if child.text is None:
                    print('Entry has no date. Skipped')
                    skipped+=1
                    continue
                try:
                   date = datetime.datetime.strptime(child.text[0:19], '%Y-%m-%d %H:%M:%S')
                except:
                    print('Invalid date. Skipped')
                    skipped +=1
                    continue
                local['year']=date.year
                local['month']=date.month
                local['day']=date.day
                local['hour']=date.hour
                local['minute']=date.minute
                local['second']=date.second
            
            #Parse grade tag
            elif child.tag == 'grade':
                if child.text is None:
                    print('Entry has no grade. skipped')
                    skipped+=1
                    continue
                local['grade'] = float(child.text)
            
            #Parse text tag
            elif child.tag == 'text':
                if child.text is None:
                    print('Entry has no text. skipped')
                    skipped+=1
                    continue
                local['text'] = text_to_word_sequence(child.text)
        
        #Parsing ok
        data.append(local)
        nbEntries+=1
        if idEntry % report_freq == 0:
            print('{} done over {} total ({:.2f}%)'\
                  .format(idEntry, len(elem.getchildren()), (idEntry/len(elem.getchildren())*100)))
    #First pass done
    print('First Pass done')
    print('{} entries collected, {} skipped'.format(nbEntries, skipped))
    
    ##Second pass, text formatting
    print('Aggregating text data')
    text_len = list(map(lambda d: len(d['text']), data))
    text_len = np.array(text_len)
    max_len = int(text_len.mean()+3*text_len.std())
    print('Average text length is: {:.2f} (std: {:.2f}) setting MAX_LEN to {}'\
          .format(text_len.mean(), text_len.std(), max_len))
    
    lexicon = reduce(typeCast.update_return, [set()] + list(map(lambda d: d['text'], data)))
    lexicon = list(lexicon)
    word_indices = dict((w, idx) for idx, w in enumerate(lexicon))
    print('Lexicon size is {}'.format(len(lexicon)))
    
    print('Formatting text data')
    report_freq = int(0.1 * len(data))
    for idData, d in enumerate(data):
        d['text'] = list(map(lambda w: word_indices[w], d['text']))
        d['text'] = sequence.pad_sequences([d['text']], maxlen=max_len)[0]
        
        if idData % report_freq == 0:
            print('{} done over {} total ({:.2f}%)'\
                  .format(idData, len(data), (idData/len(data)*100)))
            
    #All done
    return TextData(data)
        
        
    
    
        
        
    
        
        
                       
            
        
        
    

def loadXML(path, parserFunction):
    return TextData(parserFunction(path))

class TextData:
    def __init__(self, data):
        return