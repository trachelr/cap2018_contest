import platform
import os
import json

import parse_xml
import typeCast

import MapReduceData as mrd
import numpy as np
import pandas as pd

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




if __name__ == '__main__':
    #Use platform as os.uname() does not exist on Windows
    if platform.uname()[1] == 'DESKTOP-42C7TJ2':
        file_path = 'data/Cambridge_cleaned.xml'
        parsed_path = 'data/Cambridge_parsed.json'
        dFrame_path = 'data/dataset.csv'
    else:
        file_path = 'EF201403_selection121.xml'
        parsed_path = '' #TODO set a parsed path
        dFrame_path = '' #TODO set a dataframe path
        
        
    if not os.path.isfile(dFrame_path):
        if not os.path.isfile(parsed_path):
            data = parse_xml.parseXML(file_path, False)
            print('Saving data')
            fp = open(parsed_path, 'w')
            json.dump(data, fp, cls=typeCast.NumpyEncoder)
            fp.close()
            print('Done')
            data = mrd.MapReduceData(data)
        else:
            print('Loading data')
            fp = open(parsed_path, 'r')
            data = json.load(fp)
            fp.close()
            print('Done')
            #Json serialization break np.array into list. But lists tends to be less memory efficient.
            #Thus we recreate the array structure first.
            #In practice, this is mandatory to prevent OOM errors.
            print('Post load restructuration')
            data = mrd.MapReduceData(data)
            data = data.map(lambda x: dict( (k, np.array(v) if k == 'text' else v) for k, v in x.items()))
            print('Done')
        
        print('Creating Dataset')
        data = data.selectKeys(['entry_id', 'level', 'unit', 'learner_id', 'nationality', 'topic_id', 'grade', 'text'])
        fn, ds = data.getDataSet()
        print('Done')
        
        print('Saving Dataset')
        df = pd.DataFrame(data=ds, columns=fn)
        df.to_csv(dFrame_path, sep=';')
        print('Done')
    else:
        df = pd.read_csv(dFrame_path)
        
        
    assert False
    
    
    
        
    
    
#    sentences, grades, levels, learners, indexes = [], [], [], [], []
#    s = ''
#    for event, elem in etree.iterparse(file_path,
#                                       html=True, events=('start', 'start-ns', 'end', 'end-ns')):
#        if elem.tag == 'writing':
#            elems = dict(elem.items())
#            level = int(elems['level'])
#            sample_id = int(elems['id'])
#            print('Parsing sample', sample_id)
#            for subelem in elem.getchildren():
#                if subelem.tag == 'learner':
#                    for key, val in subelem.items():
#                        if key == 'id':
#                            learn_id = int(val)
#                elif subelem.tag == 'grade':
#                    if subelem.text is not None:
#                        grade = float(subelem.text)
#                    else:
#                        continue
#                elif subelem.tag == 'text':
#                    if subelem.text is not None:
#                        # convert the text and filter out characters
#                        text_seq = text_to_word_sequence(subelem.text)
#                        s += subelem.text
#                    else:
#                        continue
#    
#            # append data if parsing went well...
#            sentences.append(text_seq)
#            grades.append(grade)
#            levels.append(level)
#            learners.append(learn_id)
#            indexes.append(sample_id)
#    
#    d = set(sentences[0])
#    for si in sentences:
#        d.update(si)
#    
#    word_list = list(d)
#    word_indices = dict((c, i) for i, c in enumerate(word_list))
#    indices_word = dict((i, c) for i, c in enumerate(word_list))
#    
#    
#    MAX_LENGTH = 150
#    def blog_to_word_seq(blog):
#        blog_words = list(blog)
#        blog_words_indices = list(map(lambda char: word_indices[char], blog_words))
#        return sequence.pad_sequences([blog_words_indices], maxlen=MAX_LENGTH)[0]
    
    
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
