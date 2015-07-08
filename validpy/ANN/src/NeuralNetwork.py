__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

from pybrain.structure import *
from scipy.spatial import distance
from numpy import array


class NeuralNetwork():
    """Class containing the ANN model

    Attributes:
        network (FeedForwardNetwork): ANN model

    """
    def __init__(self, hidden_layers, units, function, input_length, output_length):
        """
        Initializes the model with the given parameters

        :param hidden_layers: number of hidden layers
        :type hidden_layers: int
        :param units: number of units
        :type units: int
        :param function: network's activation function (possible values: linear, sigmoid, gaussian)
        :type function: str
        :param input_length: input length
        :type input_length: int
        :param output_length: output length
        :type output_length: int
        :return: None
        :rtype: None
        """
        self.network = FeedForwardNetwork()
        layer = initialize_layer('linear', input_length)
        self.network.addInputModule(layer)
        for layer_info in range(hidden_layers):
            prev_layer = layer
            layer = initialize_layer(function, units)
            self.network.addModule(layer)
            conn = initialize_connection('full', prev_layer, layer)
            self.network.addConnection(conn)
        prev_layer = layer
        layer = initialize_layer('linear', output_length)
        self.network.addOutputModule(layer)
        conn = initialize_connection('full', prev_layer, layer)
        self.network.addConnection(conn)

        # Sorting topographically the modules and return the network
        self.network.sortModules()

    def valid(self, validation_set):
        """
        Compute the average euclidean distance activating
        the model over a validation set

        :param validation_set: validation set
        :type validation_set: SupervisedDataSet
        :return: average euclidean distance
        :rtype: float
        """
        dist = 0
        for example in validation_set:
            res = self.network.activate(example[0])
            res = array(res)
            target = array(example[1])
            dist += distance.euclidean(res, target)
        return dist/len(validation_set)


def initialize_layer(function, length):
    """
    Initialize a layer with the given parameters

    :param function: layer's activation function (possible values: linear, sigmoid, gaussian)
    :type function: string
    :param length:  layer's length
    :type length: int
    :return: layer
    :rtype: NeuronLayer
    """
    layer = None
    if function == "linear":
        layer = LinearLayer(length)
    elif function == "sigmoid":
        layer = SigmoidLayer(length)
    elif function == "gaussian":
        layer = GaussianLayer(length)
    return layer


def initialize_connection(connection, prev_layer, layer):
    """
    Initialize a connection between two layers

    :param connection: layer's connection topology (possible values: full)
    :type connection: string
    :param prev_layer: first layer
    :type prev_layer: NeuronLayer
    :param layer: second Layer
    :type layer: NeuronLayer
    :return: connection
    :rtype: Connection
    """
    conn = None
    if connection == "full":
        conn = FullConnection(prev_layer, layer)
    return conn