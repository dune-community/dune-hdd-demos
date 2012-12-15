#!/bin/bash

# always assume that we are in local/bin/
BASEDIR=$(cd $(dirname ${0}) && cd ../.. && pwd)
cd $BASEDIR \
  && test -f PATH \
  && source PATH

echo -n "building demos... "
mkdir -p $BASEDIR/demos \
  ; cd $BASEDIR/demos \
  && cmake ../dune-detailed-solvers &> /dev/null \
  && make examples_stationary_linear_elliptic_cg_detailed_discretizations &> /dev/null \
  && echo "done!" \
  && echo "go to $BASEDIR/demos and run one of the following:" \
  && echo "  cd examples/stationary/linear/elliptic/cg/ && ./examples_stationary_linear_elliptic_cg_detailed_discretizations" \
  || echo "failed!"

exit 0
