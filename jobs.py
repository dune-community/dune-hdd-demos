#!/usr/bin/env python2

from __future__ import print_function

import sys

from OS2014_main import dune_config, config, init_dune, create_discretization, CubicParameterSpace, run_experiment


dune_num_refinements = [1, 2, 3, 4]
num_training_samples = [10, 25, 50, 100, 200]
greedy_use_estimator = [True, False]

product = ('elliptic', 'penalty')
norm = 'elliptic'


tmp_cfg = config.copy()
for kk, vv in dune_config.items():
    assert not tmp_cfg.has_key(kk)
    tmp_cfg[kk] = vv

configs = []
for level in dune_num_refinements:
    for sample in num_training_samples:
        for use in greedy_use_estimator:
            cfg = tmp_cfg.copy()
            cfg['dune_num_refinements'] = level
            cfg['num_training_samples'] = sample
            cfg['greedy_use_estimator'] = use
            configs.append(cfg)

assert len(sys.argv) == 2
job_number = int(sys.argv[1])
assert job_number >= 0
assert job_number < len(configs)


print('running job {} with config:'.format(job_number))
print('')
for kk, vv in configs[job_number].items():
    print('{}: {}'.format(kk, vv))
print('')
print('initializing dune module... ')
example, wrapper = init_dune(configs[job_number])
discretization = create_discretization(example, wrapper, configs[job_number])
discretization = discretization.with_(parameter_space=CubicParameterSpace(discretization.parameter_type, 0.1, 1.0))
print('the discretization has {} DoFs.'.format(discretization.solution_space.dim))
print('the parameter type is {}.'.format(discretization.parameter_type))
print('')
print('running experiment with product \'{}\' and norm \'{}\':'.format(product, norm))
run_experiment(example, wrapper, discretization, configs[job_number], product, norm)

