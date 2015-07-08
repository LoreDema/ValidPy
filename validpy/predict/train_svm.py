__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

import sys
import os
import codecs
import simplejson
from sklearn.svm import SVR
import pickle


def main(conf_file, separator=','):
    """
    Train a model

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

    svr = []
    for j in range(conf.get('output_length')):
        svr.append('svr' + str(j) + '.pickle')

    outs = []
    x = [example[0] for example in examples]
    for i, sv in enumerate(svr):
        y = []
        for example in examples:
            y.append(example[1][i])
        outs.append(y)

    clf = []
    for i, sv in enumerate(svr):
        clf.append(SVR(kernel=conf.get('kernel'), C=conf.get('C'), epsilon=conf.get('epsilon'), degree=conf.get('degree')))
        clf[-1].fit(x, outs[i])
        # save svr
        pickle.dump(clf[-1], open(conf.get('out_folder') + sv, 'wb'))


if __name__ == '__main__':
    main(sys.argv[1])
