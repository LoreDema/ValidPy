#!/bin/sh
DIR=$( readlink -f "$( dirname "$0" )")
python -c 'import validpy.ANN.src.experimentKcross as ex; ex.main("'$DIR'/'$1'")'
