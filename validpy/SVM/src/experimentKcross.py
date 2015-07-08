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
from sklearn.svm import SVR
import pickle


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


def validate(clfs, validation_set):
    """
    Compute the average euclidean distance activating
    the model over a validation set

    :param clfs: list of model
    :type clfs: list[SVR]
    :param validation_set: Validation set
    :type validation_set: list[tuple(list[float], list[float])]
    :return: average euclidean distance
    :rtype: float
    """
    dist = 0

    for example in validation_set:
        res = []
        for clf in clfs:
            res.append(clf.predict([example[0]]))
        target = array(example[1])
        dist += distance.euclidean(res, target)
    return dist/len(validation_set)


def k_train(kernel, c, epsilon, degree, train, valid, folder, report, svr, q):
    """
    Runs the training and the model validation, this function have to been called via Process()
    https://docs.python.org/3/library/multiprocessing.html

    :param kernel: Kernel function (possible values: linear, poly, rbf, sigmoid)
    :type kernel: string
    :param c: penalty parameter C of the error term
    :type c: float
    :param epsilon: Epsilon in the epsilon-SVR model, it specifies the epsilon-tube
        within which no penalty is associated in the training loss function
        with points predicted within a distance epsilon from the actual value
    :type epsilon: float
    :param degree: Degree of kernel function is significant only in poly, rbf, sigmoid
    :type degree: int
    :param train: training set
    :type train: list[tuple(list[float], list[float])]
    :param valid: validation set
    :type valid: list[tuple(list[float], list[float])]
    :param folder: path to the output experiment folder
    :type folder: str
    :param report: path to the output experiment report file (.txt)
    :type report: str
    :param svr: list of file name where to save models
    :type svr: list[str]
    :param q: Queue
    :type q: Queue
    :return: training time and average validation error
    :rtype: list[float]
    """
    outs = []
    x = [example[0] for example in train]
    for i, sv in enumerate(svr):
        y = []
        for example in train:
            y.append(example[1][i])
        outs.append(y)

    t0 = time.time()
    clf = []
    for i, sv in enumerate(svr):
        clf.append(SVR(kernel=kernel, C=c, epsilon=epsilon, degree=degree))
        clf[-1].fit(x, outs[i])
    training_time = time.time() - t0
    # save svr
    for i, sv in enumerate(svr):
        pickle.dump(clf[i], open(folder + sv, 'wb'))

    validation_error = validate(clf, valid)
    out_file = open(folder + report, "w")
    out_file.write("Training time: " + str(training_time) + "\nValidation error: " + str(validation_error))
    out_file.close()
    q.put([training_time, validation_error])


def k_cross_experiment(kernel, c, epsilon, degree, data_sets, out_folder, n_processes, output_length, csv=None):
    """
    Executes the k-cross validation

    :param kernel: Kernel function (possible values: linear, poly, rbf, sigmoid)
    :type kernel: string
    :param c: penalty parameter C of the error term
    :type c: float
    :param epsilon: Epsilon in the epsilon-SVR model, it specifies the epsilon-tube
        within which no penalty is associated in the training loss function
        with points predicted within a distance epsilon from the actual value
    :type epsilon: float
    :param degree: Degree of kernel function is significant only in poly, rbf, sigmoid
    :type degree: int
    :param data_sets: List of the k data sets
    :type data_sets: list[list[tuple(list[float], list[float])]]
    :param out_folder: Path to the output folder
    :type out_folder: str
    :param output_length: Output length
    :type output_length: int
    :param n_processes: Number of parallel processes
    :type n_processes: int
    :param csv: Output csv file
    :type csv: FileIO
    :return: Average error and average training time over the k experiments
    :rtype: tuple(float, float)
    """

    folder = out_folder + '/kernel_' + kernel + '/C_' + str(c) + '/epsilon_' + str(epsilon) + '/degree_' + str(degree)
    os.makedirs(folder)

    queues = []
    processes = []
    res = []
    for i, data_set in enumerate(data_sets):
        train = []
        valid = data_sets[i]
        for w in data_sets[:i]:
            train = train + w
        for w in data_sets[i+1:]:
            train = train + w

        rep = '/report' + str(i+1) + '.txt'
        svr = []
        for j in range(output_length):
            svr.append('/svr' + str(i+1) + '_' + str(j) + '.pickle')

        q = Queue()
        queues.append(q)
        p = Process(target=k_train, args=(kernel, c, epsilon, degree, train, valid, folder, rep, svr, q,))
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
        csv.write(kernel + ',' + str(c) + ',' + str(epsilon) + ',' + str(degree)
                  + ',' + str(error) + ',' + str(training_time) + '\n')
    out_file = open(folder + '/report.txt', 'w')
    out_file.write("Training time: " + str(training_time) + "\nValidation error: " + str(error))
    out_file.close()

    return error, training_time


def grid(conf, data_sets, output_length):
    """
    Runs the k-cross validation over all possible parameter combinations

    :param conf: configuration JSON
    :param data_sets: examples as list of tuple (input, output)
    :type conf: JSON
    :type data_sets: list[list[tuple(list[float], list[float])]]
    :return: None
    """
    csv = open(conf.get('out_folder') + '/distances.csv', 'w')
    csv.write('kernel,C,epsilon,degree,valid result,training time\n')
    for kernel in conf.get('kernel'):
        for c in conf.get('C'):
            for epsilon in conf.get('epsilon'):
                for degree in conf.get('degree'):
                    k_cross_experiment(kernel, c, epsilon, degree, data_sets, conf.get('out_folder'),
                                       conf.get('parallel_process'), output_length, csv)
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

    if conf.get('grid'):
        grid(conf, data_sets, conf.get('output_length'))

if __name__ == '__main__':
    main(sys.argv[1])