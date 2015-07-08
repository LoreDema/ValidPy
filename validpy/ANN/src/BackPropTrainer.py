__author__ = 'Lorenzo De Mattei'
__license__ = 'GPL'
__email__ = 'lorenzo.demattei@gmail.com'

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
import matplotlib.pyplot as plt


class Trainer():
    """Backpropagation trainer

    Attributes:
        trainer (BackpropTrainer): Backpropagation trainer
        training_errors (list): List containing the training error for each epoch
        validation_errors (list): List containing the validation error for each epoch
    """
    def __init__(self, network, momentum, learning_rate, lr_decay, data_set):
        """
        Initialize the trainer with the given parameters

        :param network: model
        :type network: NeuralNetwork.NeuralNetwork
        :param momentum: momentum
        :type momentum: float
        :param learning_rate: learning rate
        :type learning_rate: float
        :param lr_decay: learning rate decay
        :type lr_decay: float
        :param data_set: training set
        :type data_set: SupervisedDataSet
        :return: None
        :rtype: None
        """
        self.trainer = BackpropTrainer(network, dataset=data_set, momentum=momentum, learningrate=learning_rate,
                                       lrdecay=lr_decay, verbose=False)
        self.training_errors = []
        self.validation_errors = []

    def train(self, network, valid_bp, path):
        """
        Train until convergence, stopping the training when the training
        doesn't reduce the validation error after 1000 continuous epochs

        :param network: model
        :type network: NeuralNetwork.NeuralNetwork
        :param valid_bp: Validation set
        :type valid_bp: SupervisedDataSet
        :param path: Path where to save the trained model
        :type path: str
        :return: None
        :rtype: None
        """
        epochs = 0
        continue_epochs = 0
        # best_epoch = 0
        NetworkWriter.writeToFile(network.network, path)
        min_error = network.valid(valid_bp)
        while True:
            train_error = self.trainer.train()
            valid_error = network.valid(valid_bp)
            if valid_error < min_error:
                min_error = valid_error
                # best_epoch = epochs
                NetworkWriter.writeToFile(network.network, path)
                continue_epochs = 0
            self.training_errors.append(train_error)
            self.validation_errors.append(valid_error)
            epochs += 1
            continue_epochs += 1
            # print str(epochs) + " " + str(continue_epochs) + " " + str(best_epoch)
            if continue_epochs > 1000:
                break

    def plot_epochs(self, folder):
        """
        Plot the training and validation errors over the epochs

        :param folder: path to the plot file (.png)
        :type folder: str
        :return: None
        :rtype: None
        """
        plt.plot(range(len(self.validation_errors)), self.validation_errors, label='Validation errors', color='b')
        plt.plot(range(len(self.training_errors)), self.training_errors,  label='training_errors', color='r',)
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.legend()
        plt.savefig(folder)