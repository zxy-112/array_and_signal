import os
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
import aray
import signl
from utils import *

SAVE_PATH = os.path.join(os.path.expanduser('~'), 'coherent_simu_results')
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

BASE_UNIFORM_SETTINGS = {
        'signals': [],
        'element_number': 8,
        'length': float('inf'),
        'reference_position': 0,
        'interval': 0.5,
        'sample_points': 1000
        }

BASE_SIGNAL_SETTINGS = {
        'theta': 0,
        'signal_length': 10e-6,
        'amplitude': 1,
        'fre_shift': 0,
        'phase_shift': 0,
        'delay': 0,
        'signal_type': 'coherent_interference'
        }

def make_config_maker(reference_dic):
    def config_maker(update_dict):
        new_config = reference_dic.copy()
        new_config.update(update_dict)
        return new_config
    return config_maker
ary_config_maker = make_config_maker(BASE_UNIFORM_SETTINGS)
signal_config_maker = make_config_maker(BASE_SIGNAL_SETTINGS)

def make_uniform_array(update_dict):
    return aray.UniformLineArray.with_settings(ary_config_maker(update_dict))

def make_signal_maker(signal_type):
    def signal_maker(update_dict):
        return signal_type(**(signal_config_maker(update_dict)))
    return signal_maker

lfm2D_maker = make_signal_maker(signl.LfmWave2D)
noise2D_maker = make_signal_maker(signl.NoiseWave2D)
cos2D_maker = make_signal_maker(signl.CosWave2D)

signals = [
        lfm2D_maker({'signal_type': 'expect'}),
        lfm2D_maker({'theta': 20, 'signal_type': 'coherent_interference', 'amplitude': 10}),
        cos2D_maker({'signal_type': 'interference', 'theta': -20, 'amplitude': 10})
        ]
ary = make_uniform_array({'signals': signals})

output = ary.output
cov_mat = calcu_cov(output)
weight = ary.steer_vector(0)
adaptive_weight = mvdr(output, weight)

my_plot(output[0].real)

synoutput = syn(weight, output)
my_plot(synoutput.real)

ary.response_plot(weight)
ary.response_plot(adaptive_weight)

synoutput = syn(adaptive_weight, output)
my_plot(synoutput.real)

