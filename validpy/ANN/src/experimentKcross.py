__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

import sys
import codecs
import simplejson
import numpy as np
import random
import time
import os
from multiprocessing import Process, Queue
from numpy import array
from scipy.spatial import distance

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.xml.networkreader import NetworkReader

from NeuralNetwork import NeuralNetwork
from BackPropTrainer import Trainer


def split_data_set_file(file_name, input_length, output_length, k, separator=','):
    """
    Reads the data set from csv, split the input and the output,
    shuffle the example list and split it in to k subsets

    :param file_name: path to the csv file
    :type file_name: str
    :param input_length: input length
    :type input_length: int
    :param output_length: output length
    :type output_length: int
    :param k: number of subsets
    :type k: int
    :param separator: csv file separators, default = ','
    :type separator: Optional[str]
    :return: list of subsets
    :rtype: list[list[tuple]]
    """
    examples = []
    with codecs.open(file_name) as infile:
        for line in infile:
            try:
                if not (line[0] in ["#", "\n"]):
                    values = line.split(separator)
                    if len(values) != (input_length + output_length + 1):
                        continue
                    for i, value in enumerate(values):
                        values[i] = float(value)

                    examples.append((np.array(values[1:input_length + 1]), np.array(values[input_length + 1:])))
            except IndexError:
                continue
    random.shuffle(examples)
    division = len(examples) / float(k)
    return [examples[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(k)]


def validates(net_path, validation_set):
    """
    Compute the average euclidean distance activating
    the model over a validation set

    :param net_path: Path to the model
    :type net_path: str
    :param validation_set: Validation set
    :type validation_set: list[tuple(list[float], list[float])]
    :return: average euclidean distance
    :rtype: float
    """
    net = NetworkReader.readFrom(net_path)
    dist = 0
    for example in validation_set:
            res = net.activate(example[0])
            res = array(res)
            target = array(example[1])
            dist += distance.euclidean(res, target)
    return dist/len(validation_set)


def k_train(hidden_layers, units, function, input_length, output_length, momentum, learning_rate,
            lr_decay, train, valid_bp, valid, folder, report, plot, net_path, q):
    """
    Runs the training and the model validation, this function have to been called via Process()
    https://docs.python.org/3/library/multiprocessing.html

    :param hidden_layers: number of hidden layers
    :type hidden_layers: int
    :param units: number of units for each hidden layer
    :type units: int
    :param function: network's activation function (possible values: linear, sigmoid, gaussian)
    :type function: str
    :param input_length: input length
    :type input_length: int
    :param output_length: output length
    :type output_length: int
    :param momentum: momentum value
    :type momentum: float
    :param learning_rate: learning rate value
    :type learning_rate: float
    :param lr_decay: learning rate value
    :type lr_decay: float
    :param train: training set
    :type train: list[tuple(list[float], list[float])]
    :param valid_bp: backpropagation validation set
    :type valid_bp: list[tuple(list[float], list[float])]
    :param valid: validation set
    :type valid: list[tuple(list[float], list[float])]
    :param folder: path to the output experiment folder
    :type folder: str
    :param report: path to the output experiment report file (.txt)
    :type report: str
    :param plot: path to the output experiment plot file (.png)
    :type plot: str
    :param net_path: path to the output experiment model file (.xml)
    :type net_path:
    :param q: Queue
    :type q: Queue
    :return: training time and average validation error
    :rtype: list[float]
    """
    network = NeuralNetwork(hidden_layers, units, function, input_length, output_length)
    data_set = SupervisedDataSet(input_length, output_length)
    for example in train:
        data_set.addSample(example[0], example[1])
    trainer = Trainer(network.network, momentum, learning_rate, lr_decay, data_set)
    t0 = time.time()
    trainer.train(network, valid_bp, folder + net_path)
    training_time = time.time() - t0
    trainer.plot_epochs(folder + plot)
    validation_error = validates(folder + net_path, valid)
    out_file = open(folder + report, "w")
    out_file.write("Training time: " + str(training_time) + "\nValidation error: " + str(validation_error))
    out_file.close()
    q.put([training_time, validation_error])


def k_cross_experiment(hidden_layers, units, function, learning_rate, momentum, lr_decay,
                       data_sets, out_folder, input_length, output_length, n_processes, csv=None):
    """
    Executes the k-cross validation

    :param hidden_layers: Number of hidden layers
    :type hidden_layers: int
    :param units: Number of units for each hidden layer
    :type units: int
    :param function: Network's activation function (possible values: linear, sigmoid, gaussian)
    :type function: str
    :param learning_rate: Learning rate value
    :type learning_rate: float
    :param momentum: Momentum value
    :type momentum: float
    :param lr_decay: Learning rate value
    :type lr_decay: float
    :param data_sets: List of the k data sets
    :type data_sets: list[list[tuple(list[float], list[float])]]
    :param out_folder: Path to the output folder
    :type out_folder: str
    :param input_length: Input length
    :type input_length: int
    :param output_length: Output length
    :type output_length: int
    :param n_processes: Number of parallel processes
    :type n_processes: int
    :param csv: Output csv file
    :type csv: FileIO
    :return: Average error and average training time over the k experiments
    :rtype: tuple(float, float)
    """

    folder = out_folder + '/layers_' + str(hidden_layers) + '/units_' + str(units) + '/function_' + function + \
        '/lr_' + str(learning_rate) + '/momentum_' + str(momentum) + '/lr_decay' + str(lr_decay)

    os.makedirs(folder)

    queues = []
    processes = []
    res = []
    print len(data_sets)
    for i, data_set in enumerate(data_sets):
        train = []
        if i == 0:
            valid = data_sets[0]
            valid_bp = data_sets[1]
            for w in data_sets[2:]:
                train = train + w
        elif i == len(data_sets)-1:
            valid = data_sets[i]
            valid_bp = data_sets[0]
            for w in data_sets[1:i]:
                train = train + w
        else:
            valid = data_sets[i]
            valid_bp = data_sets[i+1]
            for w in data_sets[:i]:
                train = train + w
            for w in data_sets[i+2:]:
                train = train + w

        rep = '/report' + str(i+1) + '.txt'
        plot = '/plot' + str(i+1) + '.png'
        net = '/network' + str(i+1) + '.xml'

        q = Queue()
        queues.append(q)
        p = Process(target=k_train, args=(hidden_layers, units, function, input_length,
                                          output_length, momentum, learning_rate, lr_decay,
                                          train, valid_bp, valid, folder, rep, plot, net, q,))
        processes.append(p)

        if ((i % n_processes) == (n_processes-1)) or (i == len(data_sets)-1):
            for process in processes:
                process.start()
            for queue in queues:
                res.append(queue.get())
            for process in processes:
                process.join()
            processes = []
            queues = []

    training_time = 0
    error = 0
    for x in res:
        training_time += x[0]
        error += x[1]
    training_time /= len(res)
    error /= len(res)
    if csv is not None:
        csv.write(str(hidden_layers) + ',' + str(units) + ',' + function + ',' +
                  str(learning_rate) + ',' + str(momentum) + ',' + str(lr_decay) + ',' +
                  str(error) + ',' + str(training_time) + '\n')
    out_file = open(folder + '/report.txt', 'w')
    out_file.write("Training time: " + str(training_time) + "\nValidation error: " + str(error))
    out_file.close()

    return error, training_time


def grid(conf, data_sets):
    """
    Runs the k-cross validation over all possible parameter combinations

    :param conf: configuration JSON
    :param data_sets: examples as list of tuple (input, output)
    :type conf: JSON
    :type data_sets: list[list[tuple(list[float], list[float])]]
    :return: None
    """
    csv = open(conf.get('out_folder') + '/distances.csv', 'w')
    csv.write('hidden_layers,units,function,learning_rate,momentum,lr_decay,valid result,training time\n')
    for hidden_layers in conf.get('hidden_layers'):
        for units in conf.get('units'):
            for function in conf.get('function'):
                for momentum in conf.get('momentum'):
                    for learning_rate in conf.get('learning_rate'):
                        for lr_decay in conf.get('lr_decay'):
                            k_cross_experiment(hidden_layers, units, function,
                                               learning_rate, momentum, lr_decay,
                                               data_sets, conf.get('out_folder'),
                                               conf.get('input_length'), conf.get('output_length'),
                                               conf.get('parallel_process'), csv)

    csv.close()


def main(conf_file):
    """
    Reads configuration file and runs the k-cross validation experiments

    :param conf_file: path to the configuration file
    :type conf_file: str
    :return: None
    """
    conf = open(conf_file).read()
    conf = simplejson.loads(conf)

    data_sets = split_data_set_file(conf.get('data_file'), conf.get('input_length'),
                                    conf.get('output_length'), conf.get('k'))

    print len(data_sets)

    if conf.get('grid'):
        grid(conf, data_sets)

if __name__ == '__main__':
    main(sys.argv[1])