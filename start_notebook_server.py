#!/bin/bash

if [[ ! -e PATH.sh ]] ; then
  echo "Missing PATH.sh, exiting!"
  exit 1
fi
. PATH.sh

export NOTEBOOK_PATH=$BASEDIR/notebooks
export NOTEBOOK_PORT=18881

mkdir -p $NOTEBOOK_PATH
ipython2 notebook --notebook-dir=$NOTEBOOK_PATH --port=$NOTEBOOK_PORT
