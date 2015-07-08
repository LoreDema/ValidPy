#!/bin/sh
DIR=$( readlink -f "$( dirname "$0" )")
python -c 'import validpy.predict.train_svm as ex; ex.main("'$DIR'/'$1'")'
