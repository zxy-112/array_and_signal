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
ary_config_maker = make_dic_maker(BASE_UNIFORM_SETTINGS)

BASE_SIGNAL_SETTINGS = {
        'theta': 0,
        'signal_length': 10e-6,
        'amplitude': 1,
        'fre_shift': 0,
        'phase_shift': 0,
        'delay': 0,
        'signal_type': 'coherent_interference'
        }
signal_config_maker = make_dic_maker(BASE_SIGNAL_SETTINGS)

def make_uniform_array(kwargs_dict):
    return aray.UniformLineArray.with_settings(kwargs_dict)

#####################################
##change func for config maker func##
#####################################
for key in set(BASE_SIGNAL_SETTINGS.keys()) | set(BASE_UNIFORM_SETTINGS.keys()):
    exec('{}_change = lambda {}: change_maker("{}", {})'.format(*((key,)*4)))
bandwidth_change = lambda bandwidth: change_maker('bandwidth', bandwidth)

def make_add_sig_func(sig_type):
    def sig_add_change(*sig_change):
        """
        返回一个change函数，作为ary_config_maker的输入
        --sig_change: change函数，作为signal_config_maker的输入
        """
        def func(dic):
            dic['signals'].append(sig_type(**signal_config_maker(*sig_change)))
        return func
    return sig_add_change
add_lfm2D = make_add_sig_func(signl.LfmWave2D)
add_cos2D = make_add_sig_func(signl.CosWave2D)
add_noise2D = make_add_sig_func(signl.NoiseWave2D)

def add_signal(signal):
    def func(dic):
        dic['signals'].append(signal)
    return func
#########
## end ##
#########

ary = make_uniform_array(
        ary_config_maker(
            add_lfm2D(
                signal_type_change('expect')
                ),
            add_noise2D(
                signal_type_change('interference'),
                amplitude_change(10),
                theta_change(20)
                ),
            add_lfm2D(
                signal_type_change('coherent_interference'),
                theta_change(-20)
                ),
            element_number_change(16)
            )
        )
output = ary.output
cov_mat = calcu_cov(output)
weight = ary.steer_vector(0)
adaptive_weight = np.matmul(pinv_fast(cov_mat), weight)

fig, ax = plt.subplots()
ax.plot(output[0].real)
fig.show()

synoutput = np.matmul(hermitian(weight), output).squeeze()
fig, ax = plt.subplots()
ax.plot(synoutput.real)
fig.show()

ary.response_plot(weight)

cov_matrix = calcu_cov(output)
adaptive_weight = np.matmul(np.linalg.pinv(cov_matrix), weight)
ary.response_plot(adaptive_weight)

synoutput = np.matmul(hermitian(adaptive_weight), output).squeeze()
fig, ax = plt.subplots()
ax.plot(synoutput.real)
fig.show()
