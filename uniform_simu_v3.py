import os
import random
from collections import Iterable
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
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
    fig_ax[0].savefig(name, dpi=1200, transparent=True)
    plt.close(fig_ax[0])

def simulate_example():

    box_color = '#f85a40'
    line_color = '#037ef3'
    interference_color = '#7552cc'
    expect_color = '#00c16e'

    def example1():
        ary = aray_maker()
        coherent_theta = 20
        cnr = 10
        expect_theta = 0
        snr = 0
        lfm_pw = 10e-6
        sample_points = 4096
        figsize = (16, 4)
        fast_snap = 256
        plot_what = 'response'

        def make_lfm(m, n, nr):
            zero_len = lfm_pw * m
            signal_seq = []
            for _ in range(n):
                signal_seq.append(signl.ZeroSignal(zero_len/4))
                signal_seq.append(signl.Lfm(signal_length=lfm_pw, amplitude=decibel2val(nr)))
                signal_seq.append(signl.ZeroSignal(zero_len/4*3))
            return signl.SignalWave2D.concatenate(expect_theta, *signal_seq)

        e_signal = make_lfm(3, 4, snr)
        c_signal = signl.CosWave2D(theta=coherent_theta, signal_length=e_signal.signal_length, amplitude=decibel2val(cnr), signal_type='coherent_interference', fre_shift=1e6)
        c_signal = make_lfm(3, 4, snr)
        c_signal.signal_length = e_signal.signal_length
        c_signal.theta = coherent_theta

        c_signal = make_lfm(3, 4, snr)
        c_signal.signal_length = e_signal.signal_length
        c_signal.theta = -30

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

        for k in range(16):
            fig_ax_pair = plt.subplots(figsize=figsize)
            fig_ax_pair[1].plot(np.real(output[k]))
            savefig(fig_ax_pair, '阵元{}信号.png'.format(k+1))

        gs = gridspec.GridSpec(4, 4, None, 0.05, 0.05, 0.95, 0.95)
        fig_for_ani = plt.figure(figsize=(16, 8))
        fig = fig_for_ani
        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[1, :3])
        ax3 = fig.add_subplot(gs[2, :3])
        ax4 = fig.add_subplot(gs[3, :3])
        ax4.set_ylim((-1, 1))
        y_max = ax4.get_ylim()[0]
        ax5 = fig.add_subplot(gs[0, 3])
        ax6 = fig.add_subplot(gs[1, 3])
        ax7 = fig.add_subplot(gs[2, 3])
        for ax in ax5, ax6:
            ax.set_xlim((-0.4, 0.4))
            ax.set_ylim((0, 1))
        ax8 = fig.add_subplot(gs[3, 3])
        if plot_what == 'response':
            ax8.set_xlim((-90, 90))
            ax8.set_ylim((-50, 0))
            ax8.axvline(coherent_theta, color=interference_color)
            ax8.axvline(expect_theta, color=expect_color)
            ax7.set_xlim((-90, 90))
            ax7.set_ylim((-np.pi, np.pi))
            ax7.axvline(coherent_theta, color=interference_color)
            ax7.axvline(expect_theta, color=expect_color)
        else:
            ax8.set_xlim((-0.4, 0.4))
            ax8.set_ylim((0, 1))
            ax7.set_xlim((-0.4, 0.4))
            ax7.set_ylim((0, 1))

        ((line5,), (line6,), (line7,), (line8,)) = ax5.plot([], [], color=line_color), ax6.plot([], [], color=line_color), ax7.plot([], [], color=line_color), ax8.plot([], [], color=line_color),

        ax1.sharex(ax2)
        ax2.sharex(ax3)
        ax3.sharex(ax4)

        e_signal.plot(fig_ax_pair=(fig, ax1), color=line_color)
        c_signal.plot(fig_ax_pair=(fig, ax2), color=line_color)
        ax3.plot(np.real(output[0]), color=line_color)

        ax1_left = ax1.axvline(1-fast_snap, color=box_color)
        ax1_right = ax1.axvline(0, color=box_color)
        ax2_left = ax2.axvline(1-fast_snap, color=box_color)
        ax2_right = ax2.axvline(0, color=box_color)
        ax3_left = ax3.axvline(1-fast_snap, color=box_color)
        ax3_right = ax3.axvline(0, color=box_color)
        ax4_left = ax4.axvline(1-fast_snap, color=box_color)
        ax4_right = ax4.axvline(0, color=box_color)
        x_data, y_data = [], []
        ani_out_line, = ax4.plot(x_data, y_data, color=line_color)

        def ani_func(num):
            nonlocal y_max

            ax1_left.set_xdata([num+1-fast_snap, num+1-fast_snap])
            ax1_right.set_xdata([num, num])
            ax2_left.set_xdata([num+1-fast_snap, num+1-fast_snap])
            ax2_right.set_xdata([num, num])
            ax3_left.set_xdata([num+1-fast_snap, num+1-fast_snap])
            ax3_right.set_xdata([num, num])
            ax4_left.set_xdata([num+1-fast_snap, num+1-fast_snap])
            ax4_right.set_xdata([num, num])

            if num < fast_snap-1:
                used_out = np.concatenate((np.zeros((16, fast_snap-num-1), dtype=np.complex128), output[:, :num+1]), axis=1)
                used_e = np.concatenate((np.zeros(fast_snap-num-1, dtype=np.complex128), e_signal.signal[:num+1]))
                used_c = np.concatenate((np.zeros(fast_snap-num-1, dtype=np.complex128), c_signal.signal[:num+1]))
                used_single_way = np.concatenate((np.zeros(fast_snap-num-1, dtype=np.complex128), output[0][:num+1]))
            else:
                used_out = output[:, num-fast_snap+1: num+1]
                used_e = e_signal.signal[num-fast_snap+1: num+1]
                used_c = c_signal.signal[num-fast_snap+1: num+1]
                used_single_way = output[0][num-fast_snap+1: num+1]
            mvdr_weight = mvdr(used_out, expect_theta, ary.steer_vector)
            used_weight = mvdr_weight
            syn_out = syn(used_weight, used_out)
            line5.set_xdata(fftfreq(len(used_e))), line5.set_ydata(normalize(np.abs(fft(used_e))))
            line6.set_xdata(fftfreq(len(used_c))), line6.set_ydata(normalize(np.abs(fft(used_c))))
            if plot_what == 'response':
                response, thetas = ary.response_with(-90, 90, 1801, used_weight, rettheta=True)
                response = value_to_decibel(np.abs(response))
                line7.set_xdata(thetas), line7.set_ydata(np.angle(response))
                line8.set_xdata(thetas), line8.set_ydata(response)
            else:
                line7.set_xdata(fftfreq(len(used_single_way))), line7.set_ydata(normalize(np.abs(fft(used_single_way))))
                line8.set_xdata(fftfreq(len(syn_out))), line8.set_ydata(normalize(np.abs(fft(syn_out))))
            to_append = np.real(syn_out[-1])
            if to_append > y_max and num > fast_snap:
                y_max = to_append
                ax4.set_ylim((-y_max, y_max))
                ax4.figure.canvas.draw_idle()
            y_data.append(to_append)
            x_data.append(num)

            ani_out_line.set_ydata(y_data)
            ani_out_line.set_xdata(x_data)

        # ani = animation.FuncAnimation(fig, ani_func, frames=sample_points, interval=2)
        # plt.show()
        # ani.save('mvdr输入和输出.mp4', dpi=200, fps=60)

    # example1()

    def example2():
        ele_num = 16
        coherent_theta = (20,)
        incoherent_theta = (-30,)
        expect_theta = 0
        signals = list(signals_maker(coherent_theta=coherent_theta, incoherent_theta=incoherent_theta))
        sample_points = 256
        ary = aray_maker(ele_num)
        ary_for_plot = aray_maker(ele_num-1)
        ary_for_chi = aray_maker(ele_num-len(coherent_theta))
        ary_for_eval_sinr = aray_maker(ele_num)
        ary_for_eval_sinr.noise_power = 0
        ary_for_eval_sinr.receive_signal(signals[0])
        ary_for_eval_sinr.sample(sample_points)
        e_signal = ary_for_eval_sinr.output
        ary_for_eval_sinr.remove_all_signal()
        ary_for_eval_sinr.receive_signal(signals[1])
        ary_for_eval_sinr.sample(sample_points)
        c_signal = ary_for_eval_sinr.output
        ary_for_eval_sinr.remove_all_signal()
        ary_for_eval_sinr.receive_signal(signals[2])
        ary_for_eval_sinr.sample(sample_points)
        i_signal = ary_for_eval_sinr.output

        for signal in signals:
            ary.receive_signal(signal)

        ary.sample(sample_points)
        output = ary.output

        weight = ary.steer_vector(expect_theta)
        mvdr_weight = mvdr(output, expect_theta, ary.steer_vector)
        mcmv_weight = mcmv(output, expect_theta, coherent_theta, ary.steer_vector)
        ctmv_weight = ctmv(output, expect_theta, coherent_theta, ary.steer_vector, ary.noise_power)
        ctp_weight = ctp(output, expect_theta, coherent_theta, ary.steer_vector, ary.noise_power)
        duvall_weight, duvall_output = duvall(output, expect_theta, ary.steer_vector, True)
        yang_ho_chi_weight, yang_ho_chi_output = yang_ho_chi(output, len(coherent_theta), ary.steer_vector, retoutput=True)
        smooth_weight = smooth(output, expect_theta, ary.steer_vector)
        optimal_weight = exactly(expect_theta, signals[0].amplitude, (coherent_theta[0], incoherent_theta[0]), (item.amplitude for item in signals[1:]), ary.noise_power, ary.steer_vector)

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
        methods = ['MVDR', 'MCMV', 'CTMV', 'CTP', 'optimal', 'Duvall', 'smooth', 'yang_ho_chi']
        # 响应
        plt.rc('axes', prop_cycle=custom_cycler)
        fig_ax = plt.subplots()
        weights = [mvdr_weight, mcmv_weight, ctmv_weight, ctp_weight, optimal_weight]
        for weight in weights:
            ary.response_plot(weight, fig_ax_pair=fig_ax)
        ary_for_plot.response_plot(duvall_weight, fig_ax_pair=fig_ax)
        ary_for_plot.response_plot(smooth_weight, fig_ax)
        ary_for_chi.response_plot(yang_ho_chi_weight, fig_ax_pair=fig_ax)
        fig_ax[0].legend(all_lines[:len(methods)], methods)
        ary.plot_line(fig_ax)
        savefig(fig_ax, 'all_response.png')

        # 波束形成
        # def normalize(x):
        #     return x / np.max(x)

        # weights = [mvdr_weight, mcmv_weight, ctmv_weight, ctp_weight]
        # for item, name in zip(weights, methods):
        #     my_plot(normalize(np.real(syn(item, output))), num=name)
        # my_plot(normalize(np.real(duvall_output)), num='duvall_based')
        # my_plot(normalize(np.real(yang_ho_chi_output)), num='yang_ho_chi')

        plt.rcdefaults()

        ary.remove_all_signal()
        ary.noise_power = 0
        ary.receive_signal(signals[0])
        e_output = syn(mvdr_weight, ary.output)
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        ax.plot(np.real(e_output), color=line_color)
        ary.remove_all_signal()
        ary.receive_signal(signals[1])
        c_output = syn(mvdr_weight, ary.output)
        ax.plot(np.real(c_output), color='#0ead69')
        response, thetas = ary.response_with(-90, 90, 1801, mvdr_weight, True)
        ax2.plot(thetas, value_to_decibel(np.abs(response)), color=line_color)
        ax2.axvline(expect_theta, color=expect_color)
        ax2.axvline(coherent_theta, color=interference_color)
        ax3.plot(thetas, np.angle(response), color=line_color)
        ax3.axvline(expect_theta, color=expect_color)
        ax3.axvline(coherent_theta, color=interference_color)
        savefig((fig, ax), 'mvdr_out.png')
        fig, ax = plt.subplots()
        signals[0].plot(fig_ax_pair=(fig, ax), color=line_color)
        savefig((fig, ax), '期望信号.png')

        fig, ax = plt.subplots()
        signals[1].plot(fig_ax_pair=(fig, ax), color=line_color)
        savefig((fig, ax), '干扰信号.png')

        fig_for_ani, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
        optimal_weight = exactly(expect_theta, signals[0].amplitude, (coherent_theta[0], incoherent_theta[0]), (item.amplitude for item in signals[1:]), ary.noise_power, ary.steer_vector)
        ary.response_plot(optimal_weight, (fig_for_ani, ax1), linewidth=1)
        linea = ax1.lines[0]
        ary.response_plot(optimal_weight, (fig_for_ani, ax2), linewidth=1)
        lineb = ax2.lines[0]
        # mycolor = line.get_c()
        ary.response_plot(optimal_weight, (fig_for_ani, ax3), linewidth=1)
        linec = ax3.lines[0]
        ax1.set_xlim((-90, 90))
        ax1.set_ylim((-60, 0))
        ax2.set_xlim((-90, 90))
        ax2.set_ylim((-60, 0))
        ax3.set_xlim((-90, 90))
        ax3.set_ylim((-60, 0))
        ax4.set_xlim((0, sample_points))
        x_data1, y_data1 = [], []
        x_data2, y_data2 = [], []
        x_data3, y_data3 = [], []
        line1, = ax1.plot(x_data1, y_data1, color=box_color)
        ax1.legend([linea, line1], ['optimal', 'duvall'])
        line2, = ax2.plot(x_data2, y_data2, color=box_color)
        ax2.legend([lineb, line2], ['optimal', 'duvall_based'])
        line3, = ax3.plot(x_data3, y_data3, color=box_color)
        ax3.legend([linec, line3], ['optimal', 'smooth'])
        def ani_func(num):
            used_snap = output[:, :num]
            used_e_signal = e_signal[:, :num]
            used_c_signal = c_signal[:, :num]
            used_i_signal = i_signal[:, :num]
            duvall_weight, duvall_output = duvall(used_snap, expect_theta, ary.steer_vector, True)
            yang_ho_chi_weight, yang_ho_chi_output = yang_ho_chi(used_snap, len(coherent_theta), ary.steer_vector, retoutput=True)
            smooth_weight = smooth(used_snap, expect_theta, ary.steer_vector)
            y_data1, x_data1 = ary_for_plot.response_with(-90, 90, 1801, duvall_weight, True)
            y_data2, x_data2 = ary_for_plot.response_with(-90, 90, 1801, yang_ho_chi_weight, True)
            y_data3, x_data3 = ary_for_plot.response_with(-90, 90, 1801, smooth_weight, True)
            # line.set(color=mycolor)
            line1.set_xdata(x_data1), line1.set_ydata(value_to_decibel(np.abs(y_data1)))
            line2.set_xdata(x_data2), line2.set_ydata(value_to_decibel(np.abs(y_data2)))
            line3.set_xdata(x_data3), line3.set_ydata(value_to_decibel(np.abs(y_data3)))
        ani = animation.FuncAnimation(fig_for_ani, ani_func, range(1, sample_points+1), None)
        plt.show()
    # example2()

    def example3():
        plt.rcParams['lines.linewidth'] = 2
        colors = ['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#f48924', '#ffc845', '#a51890']
        plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

        ele_num = 16
        expect_theta = 0
        coherent_theta = 20
        cnr = 10
        snr = 0
        expect = signl.LfmWave2D(expect_theta, amplitude=decibel2val(snr), signal_type='expect')
        inter = signl.CosWave2D(coherent_theta, amplitude=decibel2val(cnr), signal_type='interference', fre_shift=1e6)
        inter = signl.CossWave2D(coherent_theta, amplitude=decibel2val(cnr), signal_type='coherent_interference', fres=(-4e6, -2e6, 1e6, 2e6))
        inter = signl.NoiseWave2D(coherent_theta, amplitude=decibel2val(cnr), signal_type='interference')
        inter = signl.LfmWave2D(coherent_theta, amplitude=decibel2val(cnr), signal_type='coherent_interference')
        inters_theta = (-10, -25, 10)
        inters = []
        for theta in inters_theta:
            inters.append(signl.NoiseWave2D(theta, amplitude=decibel2val(10), signal_type='interference'))
        extra_inter = False

        ary = aray_maker(ele_num=ele_num)
        ary.noise_reproducible = False
        ary.receive_signal(expect)
        ary.receive_signal(inter)
        if extra_inter:
            for inter_ in inters:
                ary.receive_signal(inter_)
        ary_15 = aray_maker(ele_num=ele_num-1)
        ary.sample(1024)

        weight_1 = 18/2
        weight_2 = 6/5
        fig, axs = plt.subplots(2, 5, figsize=(2*weight_1, 5*weight_2), gridspec_kw=dict(hspace=0.1), constrained_layout=True)
        title_config = {'fontweight': 'bold'}

        expect.plot(fig_ax_pair=(fig, axs[0, 0]))
        axs[0, 0].set_title('expect signal', **title_config)
        expect.fft_plot(fig_ax_pair=(fig, axs[1, 0]), color='C3')
        axs[1, 0].set_title('spectrum of expect signal', **title_config)

        inter.plot(fig_ax_pair=(fig, axs[0, 1]))
        axs[0, 1].set_title('interference signal', **title_config)
        inter.fft_plot(fig_ax_pair=(fig, axs[1, 1]), color='C3')
        axs[1, 1].set_title('spectrum of interference', **title_config)

        output = ary.output
        axs[0, 2].plot(output[0].real)
        axs[0, 2].set_title('received signal', **title_config)
        axs[1, 2].plot(fftfreq(len(output[0])), normalize(np.abs(fft(output[0]))), color='C3')
        axs[1, 2].set_title('spectrum of received signal', **title_config)

        mvdr_weight = mvdr(output, expect_theta, ary.steer_vector)
        duvall_weight, duvall_output = duvall(output, expect_theta, ary.steer_vector, True)
        response, thetas = ary.response_with(-90, 90, 1801, mvdr_weight, True)
        response, thetas = ary_15.response_with(-90, 90, 1801, duvall_weight, True)
        axs[0, 4].plot(thetas, value_to_decibel(np.abs(response)))
        axs[0, 4].set_title('amplitude response', **title_config)
        axs[0, 4].set_xlabel('degree(\u00b0)')
        axs[1, 4].plot(thetas, np.angle(response))
        axs[1, 4].set_title('phase response', **title_config)
        axs[1, 4].set_xlabel('degree(\u00b0)')

        linestyle = (0, (5, 1))
        for ax in (axs[0, 4], axs[1, 4]):
            ax.axvline(coherent_theta, linestyle=linestyle, color='C1', linewidth=1, label='interference')
            ax.axvline(expect_theta, linestyle=linestyle, color='C4', linewidth=1, label='expect')

        syn_out = syn(mvdr_weight, output)
        syn_out = duvall_output
        axs[0, 3].plot(syn_out.real)
        axs[0, 3].set_title('beamformer output', **title_config)
        axs[1, 3].plot(fftfreq(len(syn_out)), normalize(np.abs(fft(syn_out))), color='C3')
        axs[1, 3].set_title('spectrum of beamfromer output', **title_config)

        for ax in axs[1, :4]:
            ax.set_xlim((-0.1, 0.1))
            ax.set_xlabel('$f$  (normalized)')

        for ax in axs[0, :4]:
            ax.set_xlabel('sample')

        for item in axs:
            for ax in item:
                plt.setp(ax.get_xticklabels(), fontsize=8)
                plt.setp(ax.get_yticklabels(), fontsize=8)

        fig.show()
        savefig((fig, axs), '非相干演示.svg')

        fig, axs = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        expect.plot(fig_ax_pair=(fig, axs[0, 0]))
        axs[0, 0].set_title('expect signal', **title_config)
        expect.fft_plot(fig_ax_pair=(fig, axs[1, 0]), color='C3')
        axs[1, 0].set_title('spectrum of expect signal', **title_config)

        axs[0, 1].plot(output[0].real)
        axs[0, 1].set_title('received signal', **title_config)
        axs[1, 1].plot(fftfreq(len(output[0])), normalize(np.abs(fft(output[0]))), color='C3')
        axs[1, 1].set_title('spectrum of received signal', **title_config)

        axs[0, 2].plot(syn_out.real)
        axs[0, 2].set_title('beamformer output', **title_config)
        axs[1, 2].plot(fftfreq(len(syn_out)), normalize(np.abs(fft(syn_out))), color='C3')
        axs[1, 2].set_title('spectrum of beamfromer output', **title_config)

        for ax in axs[1, :3]:
            ax.set_xlim((-0.1, 0.1))
            ax.set_xlabel('$f$  (normalized)')

        for ax in axs[0, :3]:
            ax.set_xlabel('sample')

        axs[0, 3].plot(thetas, value_to_decibel(np.abs(response)))
        axs[0, 3].set_title('amplitude response', **title_config)
        axs[0, 3].set_xlabel('degree(\u00b0)')
        axs[1, 3].plot(thetas, np.angle(response))
        axs[1, 3].set_title('phase response', **title_config)
        axs[1, 3].set_xlabel('degree(\u00b0)')
        linestyle = (0, (5, 1))
        for ax in (axs[0, 3], axs[1, 3]):
            ax.axvline(expect_theta, color='C4', linewidth=1, linestyle=linestyle)
            for theta in chain((coherent_theta,), inters_theta):
                ax.axvline(theta, color='C1', linewidth=1, linestyle=linestyle)

        fig.show()
        savefig((fig, axs), '多个干扰.svg')

        fig, ax = plt.subplots()
        mvdr_weight = mvdr(output, expect_theta, ary.steer_vector)
        duvall_weight = duvall(output, expect_theta, ary.steer_vector)
        smooth_weight = smooth(output, expect_theta, ary.steer_vector)
        yang_ho_chi_weight = yang_ho_chi(output, 1, ary.steer_vector)

        ary.response_plot(mvdr_weight, (fig, ax), color='C0', linestyle='-', label='MVDR')
        ary_15.response_plot(duvall_weight, (fig, ax), color='C1', linestyle=(0, (5, 1)), label='proposed')
        ary_15.response_plot(yang_ho_chi_weight, (fig, ax), color='C2', linestyle='dashed', label='method in [14]')
        ary_15.response_plot(smooth_weight, (fig, ax), color='C3', linestyle='dashdot', label='spatial smooth')
        ax.axvline(expect_theta, color='C4', linewidth=1, linestyle=(0, (5, 1)))
        ax.axvline(coherent_theta, color='C5', linewidth=1, linestyle=(0, (5, 1)))
        ax.legend()
        fig.show()
        savefig((fig, ax), '对比.svg')

        plt.rcdefaults()
    example3()

    def example4():
        expect = signl.LfmWave2D()
        coherent_1 = signl.LfmWave2D(theta=-20)
        coherent_2 = signl.LfmWave2D(theta=20)
        coherent_3 = signl.LfmWave2D(theta=-30, amplitude=10)
        aray = aray_maker()
        aray_for_plot = aray_maker(15)
        for signal in (expect, coherent_1, coherent_2):
            aray.receive_signal(signal)
        aray.sample(1024)
        output = aray.output
        duvall_weight, duvall_output = duvall(output, 0, aray.steer_vector, True)
        fig, ax = plt.subplots()
        ax.plot(np.real(duvall_output), linewidth=2)
        fig.show()
        fig, ax = aray_for_plot.response_plot(duvall_weight)
        ax.axvline(-20, linestyle=':')
        ax.axvline(20, linestyle=':')
        ax.axvline(-30, linestyle=':')
        fig.show()
    # example4()

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
            yield output, np.linalg.pinv(real_cov), info_dic
            ary.remove_all_signal()

if __name__ == "__main__":
    simulate_example()
    # generate_flag = input('generate data?y/n\n')
    # if generate_flag == 'y':
    #     data_generate(data_generator(), SAVE_PATH)

