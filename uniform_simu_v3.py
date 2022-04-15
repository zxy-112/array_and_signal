import os
import random
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from cycler import cycler
import aray
import signl
from utils import *

SAVE_PATH = os.path.join(os.path.expanduser('~'), 'coherent_data')
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

noise_power = aray.UniformLineArray.noise_power
decibel2val = lambda x: np.sqrt(np.power(10, x / 10) * noise_power)

def signals_maker(expect_theta=0, coherent_theta=(20,), incoherent_theta=(-30,), snr=0, cnr=(10,), inr=(10,)):
    yield signl.LfmWave2D(theta=expect_theta, amplitude=decibel2val(snr))

    for theta, ratio in zip(coherent_theta, cnr):
        yield signl.LfmWave2D(theta=theta, signal_type='coherent_interference', amplitude=decibel2val(ratio))
    for theta, ratio in zip(incoherent_theta, inr):
        yield signl.CosWave2D(theta=theta, signal_type='interference', amplitude=decibel2val(ratio))

def aray_maker(ele_num=16):
    ary = aray.UniformLineArray()
    for _ in range(ele_num):
        ary.add_element(aray.Element())
    return ary

def simulate_example():
    ele_num = 16
    coherent_theta = (20,)
    incoherent_theta = (-30,)
    expect_theta = 0
    signals = list(signals_maker(coherent_theta=coherent_theta, incoherent_theta=incoherent_theta))
    sample_points = ele_num ** 2
    ary = aray_maker()
    ary_for_plot = aray_maker(ele_num-1)
    ary_for_chi = aray_maker(ele_num-len(coherent_theta))

    for signal in signals:
        ary.receive_signal(signal)

    ary.sample(sample_points)

    output = ary.output

    weight = ary.steer_vector(expect_theta)
    adaptive_weight = mvdr(output, expect_theta, ary.steer_vector)
    mcmv_weight = mcmv(output, expect_theta, coherent_theta, ary.steer_vector)
    ctmv_weight = ctmv(output, expect_theta, coherent_theta, ary.steer_vector, ary.noise_power)
    ctp_weight = ctp(output, expect_theta, coherent_theta, ary.steer_vector, ary.noise_power)
    duvall_weight, duvall_output = duvall(output, expect_theta, ary.steer_vector, True)
    yang_ho_chi_weight, yang_ho_chi_output = yang_ho_chi(output, len(coherent_theta), ary.steer_vector, retoutput=True)

    color=['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#f48924', '#ffc845', '#52565e']
    lw = [2, 2, 2, 2] * 2
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
    methods = ['MVDR', 'MCMV', 'CTMV', 'CTP', 'Duvall', 'yang_ho_chi']
    # 响应
    plt.rc('axes', prop_cycle=custom_cycler)
    fig_ax = plt.subplots()
    weights = [adaptive_weight, mcmv_weight, ctmv_weight, ctp_weight]
    for weight in weights:
        ary.response_plot(weight, fig_ax_pair=fig_ax)
    ary_for_plot.response_plot(duvall_weight, fig_ax_pair=fig_ax)
    ary_for_chi.response_plot(yang_ho_chi_weight, fig_ax_pair=fig_ax)
    fig_ax[0].legend(all_lines[:len(methods)], methods)

    # 波束形成
    def normalize(x):
        return x / np.max(x)

    weights = [adaptive_weight, mcmv_weight, ctmv_weight, ctp_weight]
    for item, name in zip(weights, methods):
        my_plot(normalize(np.real(syn(item, output))), num=name)
    my_plot(normalize(np.real(duvall_output)), num='duvall_based')
    my_plot(normalize(np.real(yang_ho_chi_output)), num='yang_ho_chi')

    return fig_ax

def data_generator():
    ele_num = 16
    cnr_num = 1
    inr_num = 1
    sample_points = ele_num ** 2

    ary = aray.UniformLineArray()
    for _ in range(ele_num):
        ary.add_element(aray.Element())

    def thetas():
        current = -60
        interval = .1
        while current < 60:
            if -10 <= current <= 10:
                current = 11
            yield current
            current += interval
    thetas = list(thetas())

    def check_thetas(seq):
        for m in range(len(seq)-1):
            for n in range(m+1, len(seq)):
                if abs(seq[m] - seq[n]) < 5:
                    return False
        return True

    def theta_lis(num):
        counts = 0
        while True:
            to_yield = [random.choice(thetas) for _ in range(num)]
            counts += 1
            if check_thetas(to_yield):
                yield to_yield
                counts = 0
            elif counts > 1000:
                break

    def nr_lis(num):
        # yield tuple((np.random.uniform(5, 10) for _ in range(num)))
        while True:
            yield [10 for _ in range(num)]

    expect_theta = 0
    snr = 0
    for _, coherent_theta, incoherent_theta in zip(range(10000), theta_lis(cnr_num), theta_lis(inr_num)):
        for _, coherent_nr, incoherent_nr in zip(range(1), nr_lis(cnr_num), nr_lis(inr_num)):
            signals = []
            signals.append(signl.LfmWave2D(theta=expect_theta, amplitude=decibel2val(snr)))
            real_cov = ary.noise_power * np.eye(ary.element_number, dtype=np.complex128)
            for theta, nr in zip(coherent_theta, coherent_nr):
                amplitude = decibel2val(nr)
                signals.append(signl.LfmWave2D(theta, amplitude=amplitude))
                power = amplitude ** 2
                steer_vector = ary.steer_vector(theta)
                real_cov += power * np.matmul(steer_vector, hermitian(steer_vector))
            for theta, nr in zip(incoherent_theta, incoherent_nr):
                amplitude = decibel2val(nr)
                signals.append(signl.CosWave2D(theta, amplitude=amplitude, fre_shift=6e6))
                power = amplitude ** 2
                steer_vector = ary.steer_vector(theta)
                real_cov += power * np.matmul(steer_vector, hermitian(steer_vector))
            for signal in signals:
                ary.receive_signal(signal)
            ary.sample(sample_points)
            output = ary.output
            info_dic = {
                    'coherent_theta': coherent_theta,
                    'incoherent_theta': incoherent_theta,
                    'expect_theta': expect_theta,
                    'snr': snr,
                    'cnr': nr,
                    'inr': nr
                    }
            yield output, real_cov, info_dic
            ary.remove_all_signal()

if __name__ == "__main__":
    fig_ax = simulate_example()
    generate_flag = input('generate data?y/n\n')
    fig_ax[0].savefig('response.png', dpi=600)
    if generate_flag == 'y':
        data_generate(data_generator(), SAVE_PATH)

