import os
import random
from collections import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
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

def savefig(fig_ax, name):
    fig_ax[0].savefig(name, dpi=600, transparent=True)

def simulate_example():

    def example1():
        ary = aray_maker()
        coherent_theta = 20
        cnr = 10
        expect_theta = 0
        snr = 0
        lfm_pw = 10e-6
        sample_points = 4096
        figsize = (16, 4)
        fast_snap = 64

        def make_lfm(m, n):
            zero_len = lfm_pw * m
            signal_seq = []
            for _ in range(n):
                signal_seq.append(signl.ZeroSignal(zero_len/4))
                signal_seq.append(signl.Lfm(signal_length=lfm_pw, amplitude=decibel2val(snr)))
                signal_seq.append(signl.ZeroSignal(zero_len/4*3))
            return signl.SignalWave2D.concatenate(expect_theta, *signal_seq)

        e_signal = make_lfm(3, 4)
        c_signal = signl.CosWave2D(theta=coherent_theta, signal_length=e_signal.signal_length, amplitude=decibel2val(cnr), signal_type='coherent_interference', fre_shift=1e6)
        e_signal.sample(sample_points)
        c_signal.sample(sample_points)

        fig_ax_pair = plt.subplots(figsize=figsize)
        e_signal.plot(fig_ax_pair=fig_ax_pair)
        savefig(fig_ax_pair, '期望信号.png')

        fig_ax_pair = plt.subplots(figsize=figsize)
        e_signal.fft_plot(fig_ax_pair=fig_ax_pair)
        savefig(fig_ax_pair, '期望信号频谱.png')

        fig_ax_pair = plt.subplots(figsize=figsize)
        c_signal.plot(fig_ax_pair=fig_ax_pair)
        savefig(fig_ax_pair, '干扰信号.png')

        fig_ax_pair = plt.subplots(figsize=figsize)
        c_signal.fft_plot(fig_ax_pair=fig_ax_pair)
        savefig(fig_ax_pair, '干扰信号频谱.png')

        ary.receive_signal(c_signal)
        ary.receive_signal(e_signal)
        ary.sample(sample_points)

        output = ary.output

        fig_ax_pair = plt.subplots(figsize=figsize)
        fig_ax_pair[1].plot(np.real(output[0]))
        fig_ax_pair[0].show()
        savefig(fig_ax_pair, '阵元1信号.png')

        fig_for_ani = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        fig, (ax1, ax2, ax3) = fig_for_ani
        e_signal.plot(fig_ax_pair=(fig, ax1))
        c_signal.plot(fig_ax_pair=(fig, ax2))

        box_color = 'red'
        ax1_left = ax1.axvline(1-fast_snap, color=box_color)
        ax1_right = ax1.axvline(0, color=box_color)
        ax2_left = ax2.axvline(1-fast_snap, color=box_color)
        ax2_right = ax2.axvline(0, color=box_color)
        ax3_right = ax3.axvline(0, color=box_color)
        x_data, y_data = [], []
        ani_out_line, = ax3.plot([], y_data)
        ax3.set_ylim((-1, 1))
        y_max = ax3.get_ylim()[0]

        def ani_func(num):
            nonlocal y_max

            ax1_left.set_xdata([num+1-fast_snap, num+1-fast_snap])
            ax1_right.set_xdata([num, num])
            ax2_left.set_xdata([num+1-fast_snap, num+1-fast_snap])
            ax2_right.set_xdata([num, num])
            ax3_right.set_xdata([num, num])

            if num < fast_snap-1:
                used_out = np.concatenate((np.zeros((16, fast_snap-num-1), dtype=np.complex128), output[:, :num+1]), axis=1)
            else:
                used_out = output[:, num-fast_snap+1: num+1]
            syn_out = syn(mvdr(used_out, expect_theta, ary.steer_vector), used_out)
            to_append = np.real(syn_out[-1])
            if to_append > y_max and num > fast_snap:
                y_max = to_append
                ax3.set_ylim((-y_max, y_max))
                ax3.figure.canvas.draw_idle()
            y_data.append(to_append)
            x_data.append(num)

            ani_out_line.set_ydata(y_data)
            ani_out_line.set_xdata(x_data)
            return ax1_left, ax1_right, ax2_left, ax2_right, ax3_right, ani_out_line
        ani = animation.FuncAnimation(fig, ani_func, frames=sample_points-fast_snap+1, save_count=100, interval=2,)
        ani.save('mvdr输入和输出.mp4', dpi=600)

    example1()
    def example2():
        ele_num = 16
        coherent_theta = (20,)
        incoherent_theta = (-30,)
        expect_theta = 0
        signals = list(signals_maker(coherent_theta=coherent_theta, incoherent_theta=incoherent_theta))
        sample_points = ele_num ** 2
        ary = aray_maker()
        fig_ax = ary.response_plot(ary.steer_vector(expect_theta), linewidth=2)
        savefig(fig_ax, 'response.png')
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

        ##########################
        #######plot config########
        ##########################
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
        ##########################
        #######end config#########
        ##########################
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
        ary.plot_line(fig_ax)
        savefig(fig_ax, 'all_response.png')

        # 波束形成
        def normalize(x):
            return x / np.max(x)

        weights = [adaptive_weight, mcmv_weight, ctmv_weight, ctp_weight]
        for item, name in zip(weights, methods):
            my_plot(normalize(np.real(syn(item, output))), num=name)
        my_plot(normalize(np.real(duvall_output)), num='duvall_based')
        my_plot(normalize(np.real(yang_ho_chi_output)), num='yang_ho_chi')

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
    simulate_example()
    generate_flag = input('generate data?y/n\n')
    if generate_flag == 'y':
        data_generate(data_generator(), SAVE_PATH)

