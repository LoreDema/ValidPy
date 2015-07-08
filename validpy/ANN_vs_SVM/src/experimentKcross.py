__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

import sys
import simplejson

from validpy.ANN.src import experimentKcross as experimentANN
from validpy.SVM.src import experimentKcross as experimentSVM


def main(conf_file):
    """
    Reads configuration file and runs the k-cross validation experiments

    :param conf_file: path to the configuration file
    :type conf_file: str
    :return: None
    :rtype: None
    """
    conf = open(conf_file).read()
    conf = simplejson.loads(conf)

    results = {'ANN': [], 'SVM': []}

    for i in range(conf.get('experiments')):
        data_set = experimentANN.split_data_set_file(conf.get('data_file'), conf.get('input_length'),
                                                     conf.get('output_length'), conf.get('k'))

        results['ANN'].append(experimentANN.k_cross_experiment(conf.get('ANN').get('hidden_layers'),
                                                               conf.get('ANN').get('units'),
                                                               conf.get('ANN').get('function'),
                                                               conf.get('ANN').get('learning_rate'),
                                                               conf.get('ANN').get('momentum'),
                                                               conf.get('ANN').get('lr_decay'),
                                                               data_set,
                                                               conf.get('out_folder') + '/' + str(i + 1),
                                                               conf.get('input_length'),
                                                               conf.get('output_length'),
                                                               conf.get('parallel_process')))

        results['SVM'].append(experimentSVM.k_cross_experiment(conf.get('SVM').get('kernel'),
                                                               conf.get('SVM').get('C'),
                                                               conf.get('SVM').get('epsilon'),
                                                               conf.get('SVM').get('degree'),
                                                               data_set,
                                                               conf.get('out_folder') + '/' + str(i + 1),
                                                               conf.get('parallel_process'),
                                                               conf.get('output_length')))

    csv = open(conf.get('out_folder') + '/distances.csv', 'w')
    s = 'model,'
    for i in range(conf.get('experiments')):
        s += 'error ' + str(i+1) + ',training time ' + str(i+1) + ','
    s += 'avg error,avg training time\n'
    for model in results:
        s += model + ','
        for res in results[model]:
            s += str(res[0]) + ',' + str(res[1]) + ','
        s += str(sum([result[0] for result in results[model]])/len(results[model])) + ','
        s += str(sum([result[1] for result in results[model]])/len(results[model])) + '\n'
    csv.write(s)
    csv.close()


if __name__ == '__main__':
    main(sys.argv[1])