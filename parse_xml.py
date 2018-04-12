import datetime

import typeCast

import numpy as np

from lxml import etree
from functools import reduce

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence


def parseWritingTag(writing):
    #All writing entry should follow the same hierarchy
    ret = {}
    attr = dict(writing.items())
    ret['entry_id'] = attr['id']
    ret['level'] = attr['level']
    ret['unit'] = attr['unit']
    
    #Process child entries (text, learner, topic, date and grade)
    if len(writing.getchildren()) != 5:
        print('Entry with invalid number of children (expected 5, found {}). skipped'\
              .format(len(ret.getchildren())))
        return None
    
    for child in writing.getchildren():
        attr = dict(child.items())
        
        #Parse Learner tag
        if child.tag == 'learner':
            ret['learner_id'] = int(attr['id'])
            ret['nationality'] = attr['nationality']
        
        #Parse topic tag
        elif child.tag == 'topic':
            ret['topic_id'] = int(attr['id'])
        
        #Parse date tag
        elif child.tag == 'date':
            if child.text is None:
                print('Entry has no date. Skipped')
                return None
            try:
               date = datetime.datetime.strptime(child.text[0:19], '%Y-%m-%d %H:%M:%S')
            except:
                print('Invalid date. Skipped')
                return None
            ret['year']=date.year
            ret['month']=date.month
            ret['day']=date.day
            ret['hour']=date.hour
            ret['minute']=date.minute
            ret['second']=date.second
        
        #Parse grade tag
        elif child.tag == 'grade':
            if child.text is None:
                print('Entry has no grade. skipped')
                return None
            ret['grade'] = float(child.text)
        
        #Parse text tag
        elif child.tag == 'text':
            if child.text is None:
                print('Entry has no text. skipped')
                return None
            ret['text'] = text_to_word_sequence(child.text)
            
    return ret
    

def formatText(data):
    ##Second pass, text formatting
    print('Aggregating text data')
    text_len = list(map(lambda d: len(d['text']), data))
    text_len = np.array(text_len)
    max_len = int(text_len.mean()+3*text_len.std())
    print('Average text length is: {:.2f} (std: {:.2f}, max: {:.2f}, min: {:.2f}) setting MAX_LEN to {}'\
          .format(text_len.mean(), text_len.std(), text_len.max(), text_len.min(), max_len))
    
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
    return data
    

#Default function for parsing Cambridge data
#Will either look for writing tags in XML tree or writings (notice the 's')
#Looking for writings should be faster but if there are parsing errors, some entries
#might be missing.
def parseXML(path, robust=False):
    print('Starting first pass')
    print('Robust is {}'.format(robust))
    if robust: 
        html_arg=True
        print('Will parse writing tags in child and progress will not show')
    else: 
        html_arg=False
        print('Will parse childs of writings tag but some entries might be skipped')
        
    skipped = 0
    nbEntries = 0
    data = []
    for event, elem in etree.iterparse(path, html=html_arg):
        #Look only for the 'writings' tag and break
        if (not robust) and elem.tag=='writings':
            report_freq = int(0.1 * len(elem.getchildren()))
            for idEntry, entry in enumerate(elem.getchildren()):
                if entry.tag != 'writing':
                    print('Not a writing a entry ({} found). Skip'.format(entry.tag))
                    skipped+=1
                    continue
                #Legit entry
                local = parseWritingTag(entry)
                if local is None:
                    skipped+=1
                else:
                    #Parsing ok
                    data.append(local)
                    nbEntries+=1
                #Print progress    
                if idEntry % report_freq == 0:
                    print('{} done over {} total ({:.2f}%)'\
                          .format(idEntry, len(elem.getchildren()), 
                                  (idEntry/len(elem.getchildren())*100)))
            break
        #Look for every 'writing tag'
        elif robust and elem.tag=='writing':
            local = parseWritingTag(elem)
            if local is None:
                skipped+=1
            else:
                #Parsing ok
                data.append(local)
                nbEntries+=1
            
    #First pass done
    print('First Pass done')
    print('{} entries collected, {} skipped'.format(nbEntries, skipped))
    
    return formatText(data)
    
 