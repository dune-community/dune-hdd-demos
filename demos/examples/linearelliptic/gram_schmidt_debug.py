#!/usr/bin/env python
# This file is part of the dune-pymor project:
#   https://github.com/pyMor/dune-pymor
# Copyright Holders: Felix Albrecht, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Dune LRBMS demo.

Usage:
  multiscale-generic-sipdg_demo.py SETTINGSFILE

Arguments:
  SETTINGSFILE File that can be understood by pythons ConfigParser and by dunes ParameterTree
'''

from __future__ import absolute_import, division, print_function

config_defaults = {'framework': 'rb',
                   'training_set': 'random',
                   'num_training_samples': '100',
                   'reductor': 'generic',
                   'extension_algorithm': 'gram_schmidt',
                   'extension_algorithm_product': 'h1',
                   'greedy_error_norm': 'h1',
                   'use_estimator': 'False',
                   'max_rb_size': '100',
                   'target_error': '0.01',
                   'final_compression': 'False',
                   'compression_product': 'None',
                   'test_set': 'training',
                   'num_test_samples': '100',
                   'test_error_norm': 'h1'}

import sys
import math as m
import time
from functools import partial
from itertools import izip
import numpy as np
from docopt import docopt
from scipy.sparse import coo_matrix
from scipy.sparse import bmat as sbmat
from numpy import bmat as nbmat
import ConfigParser

import linearellipticmultiscaleexample as dune_module
from dune.pymor.core import wrap_module

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2
from pymor import defaults
from pymor.algorithms import greedy, gram_schmidt_basis_extension, pod_basis_extension, trivial_basis_extension
from pymor.playground.algorithms import greedy_lrbms
from pymor.algorithms.basisextension import block_basis_extension
from pymor.core import cache
from pymor.core.exceptions import ConfigError
from pymor.discretizations import StationaryDiscretization
from pymor.la import NumpyVectorArray
from pymor.la.basic import induced_norm
from pymor.la.blockvectorarray import BlockVectorArray
from pymor.la.pod import pod
from pymor.operators import NumpyMatrixOperator
from pymor.operators.basic import NumpyLincombMatrixOperator
from pymor.operators.block import BlockOperator
from pymor.parameters import CubicParameterSpace
from pymor.reductors import reduce_generic_rb
from pymor.reductors.basic import GenericRBReconstructor, reduce_generic_rb
from pymor.reductors.linear import reduce_stationary_affine_linear

logger = core.getLogger('pymor.main.demo')
logger.setLevel('INFO')
core.getLogger('pymor.WrappedDiscretization').setLevel('WARN')
core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('dune.pymor.discretizations').setLevel('WARN')

def load_dune_module(settings_filename):

    logger.info('initializing dune module...')
    #example = dune_module.LinearellipticMultiscaleExample__DuneALUConformGrid__lt___2__2___gt__()
    example = dune_module.LinearellipticMultiscaleExample__DuneSGrid__lt___2__2___gt__()
    example.initialize([settings_filename])
    _, wrapper = wrap_module(dune_module)
    return example, wrapper


def perform_gram_schmidt_test(config, multiscale_discretization, training_samples):

    num_subdomains = multiscale_discretization._impl.num_subdomains()

    # parse config
    # first the extension algorithm product, if needed
    extension_algorithm_id = config.get('pymor', 'extension_algorithm')
    if extension_algorithm_id in {'gram_schmidt', 'pod'}:
        extension_algorithm_product_id = config.get('pymor', 'extension_algorithm_product')
        if extension_algorithm_product_id == 'None':
            extension_algorithm_products = [None for ss in np.arange(num_subdomains)]
        else:
            extension_algorithm_products = [multiscale_discretization.local_product(ss, extension_algorithm_product_id)
                                            for ss in np.arange(num_subdomains)]
    # then the extension algorithm
    if extension_algorithm_id == 'gram_schmidt':
        extension_algorithm = [partial(gram_schmidt_basis_extension, product=extension_algorithm_products[ss])
                               for ss in np.arange(num_subdomains)]
        extension_algorithm_id += ' ({})'.format(extension_algorithm_product_id)
    elif extension_algorithm_id == 'pod':
        extension_algorithm = [partial(pod_basis_extension, product=extension_algorithm_products[ss])
                               for ss in np.arange(num_subdomains)]
        extension_algorithm_id += ' ({})'.format(extension_algorithm_product_id)
    elif extension_algorithm_id == 'trivial':
        extension_algorithm = [trivial_basis_extension for ss in np.arange(num_subdomains)]
    else:
        raise ConfigError('unknown \'pymor.extension_algorithm\' given:\'{}\''.format(extension_algorithm_id))

    reduced_basis = [multiscale_discretization.local_rhs(ss).type_source.empty(dim=multiscale_discretization.local_rhs(ss).dim_source)
                     for ss in np.arange(num_subdomains)]

    for mu in training_samples:
        print('')
        print('mu = {}'.format(mu))
        U = multiscale_discretization.solve(mu)
        for ss in np.arange(num_subdomains):
            reduced_basis[ss], _  = extension_algorithm[ss](reduced_basis[ss], U.block(ss))


if __name__ == '__main__':
    # first of all, clear the cache
    cache.clear_caches()
    # parse arguments
    args = docopt(__doc__)
    config = ConfigParser.ConfigParser(config_defaults)
    try:
        config.readfp(open(args['SETTINGSFILE']))
        assert config.has_section('pymor')
    except:
        raise ConfigError('SETTINGSFILE has to be the name of an existing file that contains a [pymor] section')
    if config.has_section('pymor.defaults'):
        float_suffixes = ['_tol', '_threshold']
        boolean_suffixes = ['_find_duplicates', '_check', '_symmetrize', '_orthonormalize', '_raise_negative',
                            'compact_print']
        int_suffixes = ['_maxiter']
        for key, value in config.items('pymor.defaults'):
            if any([len(key) >= len(suffix) and key[-len(suffix):] == suffix for suffix in float_suffixes]):
                defaults.__setattr__(key, config.getfloat('pymor.defaults', key))
            elif any([len(key) >= len(suffix) and key[-len(suffix):] == suffix for suffix in boolean_suffixes]):
                defaults.__setattr__(key, config.getboolean('pymor.defaults', key))
            elif any([len(key) >= len(suffix) and key[-len(suffix):] == suffix for suffix in int_suffixes]):
                defaults.__setattr__(key, config.getint('pymor.defaults', key))

    # load module
    example, wrapper = load_dune_module(args['SETTINGSFILE'])
    logger.info('The global grid has {} elements ({}).'.format(example.grid_size(), example.type_grid()))

    # create global cg discretization
    global_cg_discretization = wrapper[example.global_discretization()]
    global_cg_discretization = global_cg_discretization.with_(
        parameter_space=CubicParameterSpace(global_cg_discretization.parameter_type, 0.1, 10.0))
    logger.info('The global CG-FEM discretization has {} DoFs.'.format(global_cg_discretization.operator.dim_source))
    # create multiscale discretization
    multiscale_discretization = wrapper[example.multiscale_discretization()]
    multiscale_discretization = multiscale_discretization.with_(
        parameter_space=global_cg_discretization.parameter_space)
    logger.info('The multiscale grid has {} subdomains.'.format(multiscale_discretization._impl.num_subdomains()))
    logger.info('The global generic-SWIPDG (with local CG-FEM) discretization has {} DoFs.'.format(
                multiscale_discretization.operator.dim_source))
    logger.info('The parameter type is {}.'.format(global_cg_discretization.parameter_type))

    # create training set
    training_set_sampling_strategy = config.get('pymor', 'training_set')
    if training_set_sampling_strategy == 'random':
        num_training_samples = config.getint('pymor', 'num_training_samples')
        training_samples = list(global_cg_discretization.parameter_space.sample_randomly(num_training_samples))
    else:
        raise ConfigError('unknown \'training_set\' sampling strategy given: \'{}\''.format(training_set_sampling_strategy))

    # run the model reduction
    framework = config.get('pymor', 'framework')
    logger.info('running lrbms with {} subdomains:'.format(multiscale_discretization._impl.num_subdomains()))
    detailed_discretization = multiscale_discretization
    perform_gram_schmidt_test(config, detailed_discretization, training_samples)
