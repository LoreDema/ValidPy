__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

import sys
import codecs
import simplejson
import os
import pickle


def main(conf_file, separator=','):
    """
    Predict over a blind set

    :param conf_file: path to the configuration file
    :type conf_file: str
    :param separator: csv file separators, default = ','
    :type separator: Optional[str]
    :return: None
    :rtype: None
    """
    conf = open(conf_file).read()
    conf = simplejson.loads(conf)

    svr = []
    for sv in os.listdir(conf.get('out_folder')):
        svr.append(pickle.load(open(conf.get('out_folder') + sv, 'rb')))
    out_file = open(conf.get('out_file'), "w")
    with codecs.open(conf.get('blind_set')) as infile:
        for line in infile:
            try:
                if not (line[0] in ["#", "\n"]):
                    values = line.split(separator)
                    if len(values) != (conf.get('input_length') + 1):
                        continue
                    for i, value in enumerate(values):
                        values[i] = float(value)

                    x = values[1:]
                    line = line[:-1]
                    for sv in svr:
                        line += ',' + str(sv.predict(x)[0])
                    out_file.write(line + '\n')
            except IndexError:
                continue
    out_file.close()


if __name__ == '__main__':
    main(sys.argv[1])