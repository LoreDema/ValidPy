Installation
============

Clone the repository:

    .. code-block:: bash

        $ git clone https://github.com/LoreDema/ValidPy.git

then install it using pip (Linux):

    .. code-block:: bash

        $ pip install -e ./ValidPy

Dependencies
============

* `simplejson <https://pypi.python.org/pypi/simplejson/>`_  3.3.1
* `NumPy <http://www.numpy.org/>`_ 1.9.2
* `PyBrain <http://pybrain.org/>`_ 0.3
* `SciPy <http://www.scipy.org/>`_ 0.13.3
* `matplotlib <http://matplotlib.org/>`_ 1.3.1
* `scikit-learn <http://scikit-learn.org/stable/>`_ 0.15.2

Quick Start
===========

This tool implement K-cross validation for both ANN and SVM.

For all the experiments you need a csv file comma ","
separated. This file have to be 3 columns, each row is:

id, input_x, output_y

ANN k-cross validation
----------------------

To perform a k-cross validation over a file you need to
create a configuration JSON like this:
    .. code-block:: JSON

        {
          "grid":"true",
          "k":8,
          "parallel_process":4,
          "data_file":"absolute_path_to_data_file.csv",
          "out_folder":"absolute_path_output_folder",
          "input_length": 10,
          "output_length": 2,
          "hidden_layers":[1,2,3],
          "units":[15,25],
          "function":["sigmoid","gaussian"],
          "momentum":[0.0,0.9],
          "learning_rate":[0.01,0.05],
          "lr_decay":[1.0, 0.9999]
        }

Then you have to run ann_kcross.sh
in executable/ giving the
path to the configuration JSON as parameter:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh ann_kcross.sh path_to_config_JSON

The script will produce a csv file containing for each
combination of the parameters the average training time and
the average average euclidean distance (computed on the validation
set outputs) over the k experiments.
It also produce for each combination a folder with the single
experiments details and models.

SVM k-cross validation
----------------------

To perform a k-cross validation over a file you need to
create a configuration JSON like this:
    .. code-block:: JSON

        {
          "grid":"true",
          "k":8,
          "parallel_process":4,
          "data_file":"absolute_path_to_data_file.csv",
          "out_folder":"absolute_path_output_folder",
          "input_length": 10,
          "output_length": 2,
          "kernel":["linear", "poly", "rbf", "sigmoid"],
          "C":[0.1, 1.0, 10, 100],
          "epsilon":[0.01,0.05, 0.1, 0.5, 1, 5],
          "degree":[3]
        }

Then you have to run svm_kcross.sh
in executable/ giving the
path to the configuration JSON as parameter:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh svm_kcross.sh path_to_config_JSON

The script will produce a csv file containing for each
combination of the parameters the average training time and
the average average euclidean distance (computed on the validation
set outputs) over the k experiments.
It also produce for each combination a folder with the single
experiments details and models.

ANN vs SVM k-cross validation
-----------------------------

To perform a k-cross validation over a file you need to
create a configuration JSON like this, you can choose how
many time to repeat the experiment setting the experiments
parameter:
    .. code-block:: JSON

        {
          "experiments":4,
          "k":8,
          "parallel_process":4,
          "data_file":"absolute_path_to_data_file.csv",
          "out_folder":"absolute_path_output_folder",
          "input_length": 10,
          "output_length": 2,
          "ANN": {
            "hidden_layers":2,
            "units":25,
            "function":"sigmoid",
            "momentum":0.0,
            "learning_rate":0.05,
            "lr_decay":0.9999
          },
          "SVM": {
            "kernel":"rbf",
            "C":30,
            "epsilon":0.1,
            "degree":3
          }
        }

Then you have to run ann_vs_svm_kcross.sh
in executable/ giving the
path to the configuration JSON as parameter:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh ann_vs_svm_kcross.sh path_to_config_JSON

The script will produce a csv file containing for each experiment
the average training time, the average average euclidean distance
over the k experiments, the total average average training time
and the total average average euclidean distance(computed on the
validation set outputs).
It also produce for each experiment a folder with the single
experiment details and models.

ANN test
--------

To perform a test you need to
create a configuration JSON like this:
    .. code-block:: JSON

        {
          "training_set":"absolute_path_to_training_set_file.csv",
          "test_set":"absolute_path_to_test_set_file.csv",
          "out_folder":"absolute_path_output_folder",
          "input_length": 10,
          "output_length": 2,
          "hidden_layers":2,
          "valid_prop":0.1,
          "units":25,
          "function":"sigmoid",
          "momentum":0.0,
          "learning_rate":0.05,
          "lr_decay":0.9999
        }

Then you have to run ann_test.sh
in executable/ giving the
path to the configuration JSON as parameter:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh ann_test.sh path_to_config_JSON

The script will produce a txt file containing the
training time and the average euclidean distance
over the test set outputs and the experiment models.

SVM test
--------

To perform a test you need to
create a configuration JSON like this:
    .. code-block:: JSON

        {
          "training_set":"absolute_path_to_training_set_file.csv",
          "test_set":"absolute_path_to_test_set_file.csv",
          "out_folder":"absolute_path_output_folder",
          "input_length": 10,
          "output_length": 2,
          "kernel":"rbf",
          "C":30,
          "epsilon":0.1,
          "degree":3
        }

Then you have to run svm_test.sh
in executable/ giving the
path to the configuration JSON as parameter:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh svm_test.sh path_to_config_JSON

The script will produce a txt file containing the
training time and the average euclidean distance
over the test set outputs and the experiment models.

SVM predict
-----------

To predict over a blind set you need a csv file comma ","
separated. This file have to be 2 columns, each row is:

id, input_x

You have to create a configuration JSON like this:
    .. code-block:: JSON

        {
          "training_set":"absolute_path_to_training_set_file.csv",
          "test_set":"absolute_path_to_test_set_file.csv",
          "out_folder":"absolute_path_output_folder",
          "out_file":"absolute_path_output_file.csv",
          "input_length": 10,
          "output_length": 2,
          "kernel":"rbf",
          "C":10,
          "epsilon":0.1,
          "degree":3
        }

Then you have to run svm_train.sh
in executable/ giving the
path to the configuration JSON as parameter:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh svm_train.sh path_to_config_JSON

The script will produce for each output a model.

Then you have to run svm_predict.sh
in executable/ giving the
path to the configuration JSON as parameter:code-block:
    .. code-block:: bash

        $ cd ./ValidPy/executable/
        $ sh svm_predict.sh path_to_config_JSON

The script will produce a csv file containing
3 columns, each row is:

id, input_x, output_y

ANN predict
-----------

Not already implemented.
