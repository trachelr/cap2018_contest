from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.preprocessing import sequence

#import xml.etree.ElementTree as etree
from lxml import etree
sentences, grades, levels, learners, indexes = [], [], [], [], []
for event, elem in etree.iterparse('EF201403_selection121.xml',
                                    html=True, events=('start', 'start-ns')):
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
                else:
                    continue

        # append data if parsing went well...
        sentences.append(text_seq)
        grades.append(grade)
        levels.append(level)
        learners.append(learn_id)
        indexes.append(sample_id)
