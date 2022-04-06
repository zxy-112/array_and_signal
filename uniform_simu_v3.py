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

ele_num = 16
sample_points = int(10000)
coherent_theta = (20, 30, 40, 50, -15, -20, -40, -50)
cnr = 0
incoherent_theta = -30
inr = 10
expect_theta = 0
snr = 0

ary = aray.UniformLineArray()
ary_for_plot = aray.UniformLineArray()
decibel2val = lambda x: np.sqrt(np.power(10, x / 10) * ary.noise_power)
for k in range(ele_num):
    ary.add_element(aray.Element())
    if k != ele_num - 1:
        ary_for_plot .add_element(aray.Element())
signals = [signl.LfmWave2D(theta=expect_theta, amplitude=decibel2val(snr)),
        signl.LfmWave2D(theta=incoherent_theta, signal_type='interference', delay=1e-6, amplitude=decibel2val(inr))]
for theta in coherent_theta:
    signals.append(signl.LfmWave2D(theta=theta, signal_type='coherent_interference', amplitude=decibel2val(cnr)))
for signal in signals:
    ary.receive_signal(signal)
ary.sample(sample_points)

output = ary.output

weight = ary.steer_vector(expect_theta)
adaptive_weight = mvdr(output, expect_theta, ary.steer_vector)
mcmv_weight = mcmv(output, expect_theta, coherent_theta, ary.steer_vector)
ctmv_weight = ctmv(output, expect_theta, coherent_theta, ary.steer_vector, ary.noise_power)
ctp_weight = ctp(output, expect_theta, coherent_theta, ary.steer_vector, ary.noise_power)
duvall_weight = duvall(output, expect_theta, ary.steer_vector)

color=['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#f48924', '#ffc845', '#52565e']
lw = [1, 1, 1, 1] * 2
linestyle = [
        (0, ()),
        (0, (5, 1, 1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (3, 1, 1, 1, 1, 1))
        ]
linestyle = linestyle * 2
assert len(color) == len(lw) == len(linestyle)
custom_cycler = (cycler(color=color) +
                 cycler(lw=lw) +
                 cycler(linestyle=linestyle))
all_lines = [mlines.Line2D([], [], **config) for config in custom_cycler]
methods = ['MVDR', 'MCMV', 'CTMV', 'CTP', 'Duvall']
# 响应
plt.rc('axes', prop_cycle=custom_cycler)
with plt.ioff():
    fig_ax = plt.subplots()
ary.response_plot(adaptive_weight, fig_ax_pair=fig_ax)
ary.response_plot(mcmv_weight, fig_ax_pair=fig_ax)
ary.response_plot(ctmv_weight, fig_ax_pair=fig_ax)
ary.response_plot(ctp_weight, fig_ax_pair=fig_ax)
ary_for_plot.response_plot(duvall_weight, fig_ax_pair=fig_ax)
fig_ax[0].legend(all_lines[:len(methods)], methods)

# 波束形成
def normalize(x):
    return x / np.max(x)

weights = [adaptive_weight, mcmv_weight, ctmv_weight, ctp_weight]
for item in weights:
    my_plot(normalize(np.real(syn(item, output))))
new_out = np.multiply(output, np.conjugate(ary.steer_vector(expect_theta)))
duvall_output = syn(duvall_weight, new_out[:-1, :])
my_plot(normalize(np.real(duvall_output)))

