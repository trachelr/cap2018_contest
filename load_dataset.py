import platform
import datetime

import typeCast
import MapReduceData

import numpy as np

from lxml import etree
from functools import reduce

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

from keras.preprocessing.text import one_hot, text_to_word_sequence
from sklearn.cross_validation import train_test_split


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
    return MapReduceData(data)

if __name__ == '__main__':
    #Use platform as os.uname() does not exist on Windows
    if platform.uname()[1] == 'DESKTOP-42C7TJ2':
        file_path = 'data/EF201403_selection59.xml'
    else:
        file_path = 'EF201403_selection121.xml'
    
    sentences, grades, levels, learners, indexes = [], [], [], [], []
    s = ''
    for event, elem in etree.iterparse(file_path,
                                       html=True, events=('start', 'start-ns', 'end', 'end-ns')):
        if elem.tag == 'writing':
            elems = dict(elem.items())
            level = int(elems['level'])
            sample_id = int(elems['id'])
            print('Parsing sample', sample_id)
            for subelem in elem.getchildren():
                if subelem.tag == 'learner':
                    for key, val in subelem.items():
                        if key == 'id':
                            learn_id = int(val)
                elif subelem.tag == 'grade':
                    if subelem.text is not None:
                        grade = float(subelem.text)
                    else:
                        continue
                elif subelem.tag == 'text':
                    if subelem.text is not None:
                        # convert the text and filter out characters
                        text_seq = text_to_word_sequence(subelem.text)
                        s += subelem.text
                    else:
                        continue
    
            # append data if parsing went well...
            sentences.append(text_seq)
            grades.append(grade)
            levels.append(level)
            learners.append(learn_id)
            indexes.append(sample_id)
    
    d = set(sentences[0])
    for si in sentences:
        d.update(si)
    
    word_list = list(d)
    word_indices = dict((c, i) for i, c in enumerate(word_list))
    indices_word = dict((i, c) for i, c in enumerate(word_list))
    
    
    MAX_LENGTH = 150
    def blog_to_word_seq(blog):
        blog_words = list(blog)
        blog_words_indices = list(map(lambda char: word_indices[char], blog_words))
        return sequence.pad_sequences([blog_words_indices], maxlen=MAX_LENGTH)[0]
    
    
    X, y = [], []
    for n, l in zip(sentences, levels):
        X.append(blog_to_word_seq(n))
        y.append(l)
    
    X = np.array(X).astype(np.uint8)
    y = np.array(y)
    y_binary = np_utils.to_categorical(y)
    print(X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.1)
    model = Sequential()
    model.add(Embedding(len(word_list), 32, input_length=MAX_LENGTH, mask_zero=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(17))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=["accuracy"])
    
    model.fit(X_train,y_train,
              batch_size=32, nb_epoch=10,
              validation_split=0.1,
              verbose=1)
    
    model.evaluate(X_test,y_test,batch_size=32)
    
    predicted_output = model.predict(X_test,batch_size=32)
    predicted_classes = model.predict_classes(X_test, batch_size=32)
