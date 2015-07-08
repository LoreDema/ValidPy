#!/bin/sh
DIR=$( readlink -f "$( dirname "$0" )")
python -c 'import validpy.predict.predict as ex; ex.main("'$DIR'/'$1'")'
