#!/bin/sh
DIR=$( readlink -f "$( dirname "$0" )")
python -c 'import test.svm as ex; ex.main("'$DIR'/'$1'")'
