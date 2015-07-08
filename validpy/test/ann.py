__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

import sys
import os
import codecs
import simplejson
from multiprocessing import Process, Queue

from validpy.ANN.src import experimentKcross as ann


def main(conf_file, separator=','):
    """
    Reads configuration and test the model over a test set

    :param conf_file: path to the configuration file
    :type conf_file: str
    :param separator: csv file separators, default = ','
    :type separator: Optional[str]
    :return: None
    :rtype: None
    """
    examples = []
    conf = open(conf_file).read()
    conf = simplejson.loads(conf)

    os.makedirs(conf.get('out_folder'))

    with codecs.open(conf.get('training_set')) as infile:
        for line in infile:
            try:
                if not (line[0] in ["#", "\n"]):
                    values = line.split(separator)
                    if len(values) != (conf.get('input_length') + conf.get('output_length') + 1):
                        continue
                    for i, value in enumerate(values):
                        values[i] = float(value)

                    examples.append(((values[1:conf.get('input_length') + 1]),
                                     (values[conf.get('input_length') + 1:])))
            except IndexError:
                continue
    test = []
    with codecs.open(conf.get('test_set')) as infile:
        for line in infile:
            try:
                if not (line[0] in ["#", "\n"]):
                    values = line.split(separator)
                    if len(values) != (conf.get('input_length') + conf.get('output_length') + 1):
                        continue
                    for i, value in enumerate(values):
                        values[i] = float(value)

                    test.append(((values[1:conf.get('input_length') + 1]),
                                 (values[conf.get('input_length') + 1:])))
            except IndexError:
                continue
    train = examples[int(round(len(examples)*conf.get('valid_prop'))):]
    valid = examples[:int(round(len(examples)*conf.get('valid_prop')))]

    q = Queue()
    p = Process(target=ann.k_train, args=(conf.get('hidden_layers'), conf.get('units'), conf.get('function'),
                                          conf.get('input_length'), conf.get('output_length'), conf.get('momentum'),
                                          conf.get('learning_rate'), conf.get('lr_decay'), train, valid, test,
                                          conf.get('out_folder'), 'report.txt', 'plot.png', 'ann.xml', q,))

    p.start()
    q.get()
    p.join()

if __name__ == '__main__':
    main(sys.argv[1])