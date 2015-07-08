#!/bin/sh
DIR=$( readlink -f "$( dirname "$0" )")
python -c 'import validpy.test.ann as ex; ex.main("'$DIR'/'$1'")'
