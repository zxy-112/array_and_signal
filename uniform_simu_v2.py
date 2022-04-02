import os
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from cycler import cycler
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
        lfm2D_maker({'signal_type': 'expect', 'band_width': 10e6}),
        lfm2D_maker({'theta': 10, 'signal_type': 'coherent_interference', 'amplitude': 10, 'band_width': 10e6}),
        cos2D_maker({'signal_type': 'interference',
            'fre_shift': 6e6,
            'theta': -20,
            'amplitude': 10})
        ]
ary = make_uniform_array({'signals': signals, 'element_number': 16})

output = ary.output

weight = ary.steer_vector(0)
adaptive_weight = mvdr(output, 0, ary.steer_vector)
mcmv_weight = mcmv(output, 0, 20, ary.steer_vector)
ctmv_weight = ctmv(output, 0, 20, ary.steer_vector, ary.noise_power)
ctp_weight = ctp(output, 0, 20, ary.steer_vector, ary.noise_power)

color=['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#f48924', '#ffc845', '#52565e']
lw = [1, 2, 2, 2] * 2
linestyle = [
        (0, ()),
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (3, 1, 1, 1, 1, 1))
        ]
linestyle = linestyle * 2
assert len(color) == len(lw) == len(linestyle)
custom_cycler = (cycler(color=color) +
                 cycler(lw=lw) +
                 cycler(linestyle=linestyle))
all_lines = [mlines.Line2D([], [], **config) for config in custom_cycler]
methods = ['MVDR', 'MCMV', 'CTMV', 'CTP']
# 响应
plt.rc('axes', prop_cycle=custom_cycler)
with plt.ioff():
    fig_ax = plt.subplots()
ary.response_plot(adaptive_weight, fig_ax_pair=fig_ax)
ary.response_plot(mcmv_weight, fig_ax_pair=fig_ax)
ary.response_plot(ctmv_weight, fig_ax_pair=fig_ax)
ary.response_plot(ctp_weight, fig_ax_pair=fig_ax)
fig_ax[0].legend(all_lines[:len(methods)], methods)

# 波束形成
def normalize(x):
    return x / np.max(x)
synoutput = syn(weight, output)
my_plot(normalize(synoutput.real))

synoutput = syn(adaptive_weight, output)
my_plot(normalize(synoutput.real))

synoutput = syn(mcmv_weight, output)
my_plot(normalize(synoutput.real))

synoutput = syn(ctmv_weight, output)
my_plot(normalize(synoutput.real))

synoutput = syn(ctp_weight, output)
my_plot(normalize(synoutput.real))

