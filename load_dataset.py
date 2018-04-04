import xml.etree.ElementTree as etree
data = []
for event, elem in etree.iterparse('EF201403_selection121.xml', events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start':
        if elem.tag == 'writing':
            new_sample = dict(elem.items())
            for subelem in elem.getchildren():
                if subelem.tag == 'learner':
                    for key, val in subelem.items():
                        if key == 'id':
                            new_sample['learn_id'] = val
                elif subelem.tag == 'grade':
                    if subelem.text is not None:
                        new_sample['grade'] = float(subelem.text)
                        continue
                elif subelem.tag == 'text':
                    new_sample['text'] = subelem.text
            data.append(new_sample)
