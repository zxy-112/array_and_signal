import os
import random
from itertools import chain
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.figure as figure
from cycler import cycler
import aray
import signl
from utils import *

colors = ['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#52565e', '#ffc845', '#a51890']
linestyles = [
    "solid",
    (0, (1, 1)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1))
]
mcmvDOAError = 0.5

SAVE_PATH = os.path.join(os.path.expanduser('~'), 'coherent_data')
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

noise_power = aray.UniformLineArray.noise_power
decibel2val = lambda x: np.sqrt(np.power(10, x / 10) * noise_power)

def signals_maker(expect_theta=0, coherent_theta=(20,), incoherent_theta=(-30,), snr=0, cnr=(10,), inr=(10,)):
    # signal generator, yield expect signal, coherent interference, sinsoid respectively.
    yield signl.LfmWave2D(theta=expect_theta, amplitude=decibel2val(snr))

    for theta, ratio in zip(coherent_theta, cnr):
        yield signl.LfmWave2D(theta=theta, signal_type='coherent_interference', amplitude=decibel2val(ratio))
    for theta, ratio in zip(incoherent_theta, inr):
        yield signl.CosWave2D(theta=theta, signal_type='interference', amplitude=decibel2val(ratio))

def aray_maker(ele_number=16, interval=0.5):
    """
    return uniform array with ele_num elements.
    """
    ary = aray.UniformLineArray(interval=interval)
    for _ in range(ele_number):
        ary.add_element(aray.Element())
    return ary

def savefig(fig, name):
    fig.savefig(os.path.join(SAVE_PATH, name), dpi=1200, transparent=True)
    plt.close(fig)

def outputWithDifferentMethod():

    ary: aray.UniformLineArray
    ary = aray_maker(ele_number=16, interval=0.5)
    coherentTheta = 20
    cnr = 10
    expectTheta = 10
    snr = 0
    samplePoints = 4000
    lfm_pw = 10e-6
    def make_lfm(m, n, nr):
        zero_len = lfm_pw * m
        signal_seq = []
        for _ in range(n):
            signal_seq.append(signl.ZeroSignal(zero_len/4))
            signal_seq.append(signl.Lfm(signal_length=lfm_pw, amplitude=decibel2val(nr)))
            signal_seq.append(signl.ZeroSignal(zero_len/4*3))
        return signl.SignalWave2D.concatenate(expectTheta, *signal_seq)
    
    expectSignal = make_lfm(1, 2, snr)
    coherentSignal = deepcopy(expectSignal)
    coherentSignal.theta = coherentTheta
    coherentSignal.amplitude = decibel2val(cnr)

    ary.receive_signal(expectSignal)
    ary.receive_signal(coherentSignal)
    ary.sample(samplePoints)
    output = ary.output

    mvdrWeight, mvdrOutput = mvdr(output, expectTheta, ary.steer_vector, True)
    mcmvWeight, mcmvOutput = mcmv(output, expectTheta, coherentTheta + mcmvDOAError, ary.steer_vector, True)
    smoothWeight, smoothOutput = smooth(output, expectTheta, ary.steer_vector, True)
    smooth8Weight, smooth8Output = smooth2(output, expectTheta, 8, ary.steer_vector, True)
    yangWeight, yangOutput = yang_ho_chi(output, 1, ary.steer_vector, expectTheta, True)
    proposedWeight, proposedOutput = proposed(output, expectTheta, ary.steer_vector, True)
    beamformOutputs = (mvdrOutput, mcmvOutput, smoothOutput, smooth8Output, yangOutput, proposedOutput)
    names = ("mvdrout", "mcmvout", "smoothout", "smooth8out", "yangout", "proposedout")
    for beamformOutput, figName in zip(beamformOutputs, names):
        fig, ax = plt.subplots()
        ax.plot(np.real(beamformOutput))
        savefig(fig, figName + ".svg")
                                    
def beamformAnimation():
    ary: aray.UniformLineArray
    figForAni: figure.Figure
    ary = aray_maker(ele_number=16, interval=0.5)
    coherentTheta = 20
    cnr = 10
    expectTheta = 10
    snr = 0
    samplePoints = 4000
    lfm_pw = 10e-6
    fastSnapshots = 128
    plotThetaNumbers = 1801
    def make_lfm(m, n, nr):
        zero_len = lfm_pw * m
        signal_seq = []
        for _ in range(n):
            signal_seq.append(signl.ZeroSignal(zero_len/4))
            signal_seq.append(signl.Lfm(signal_length=lfm_pw, amplitude=decibel2val(nr)))
            signal_seq.append(signl.ZeroSignal(zero_len/4*3))
        return signl.SignalWave2D.concatenate(expectTheta, *signal_seq)
    
    expectSignal = make_lfm(1, 2, snr)
    coherentSignal = deepcopy(expectSignal)
    coherentSignal.theta = coherentTheta
    coherentSignal.amplitude = decibel2val(cnr)

    ary.receive_signal(expectSignal)
    ary.receive_signal(coherentSignal)
    ary.sample(samplePoints)
    output = ary.output

    mvdrWeight, mvdrOutput = mvdr(output, expectTheta, ary.steer_vector, True)
    mcmvWeight, mcmvOutput = mcmv(output, expectTheta, coherentTheta + mcmvDOAError, ary.steer_vector, True)
    smoothWeight, smoothOutput = smooth(output, expectTheta, ary.steer_vector, True)
    smooth8Weight, smooth8Output = smooth2(output, expectTheta, 8, ary.steer_vector, True)
    yangWeight, yangOutput = yang_ho_chi(output, 1, ary.steer_vector, expectTheta, True)
    proposedWeight, proposedOutput = proposed(output, expectTheta, ary.steer_vector, True)

    # create axes for plot
    figForAni = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = gridspec.GridSpec(4, 4, figForAni)
    axExpect = figForAni.add_subplot(gs[0, :2])   
    axInterference = figForAni.add_subplot(gs[1, :2])
    axBeforeBeamform = figForAni.add_subplot(gs[2, :2])
    axAfterBeamform = figForAni.add_subplot(gs[3, :2])
    axExpectSpectrum = figForAni.add_subplot(gs[0, 2])
    axInterferenceSpectrum = figForAni.add_subplot(gs[1, 2])
    axBeforeBeamformSpectrum = figForAni.add_subplot(gs[2, 2])
    axAfterBeamformSpectrum = figForAni.add_subplot(gs[3, 2])
    axAmplitudeResponse = figForAni.add_subplot(gs[2, 3])
    axPhaseResponse = figForAni.add_subplot(gs[3, 3])
    axsForSpectrum = (axExpectSpectrum, axInterferenceSpectrum, axBeforeBeamformSpectrum, axAfterBeamformSpectrum)
    axsForResponse = (axPhaseResponse, axAmplitudeResponse)

    # add title
    axExpect.set_title("expect signal", fontweight="bold")
    axInterference.set_title("interference signal", fontweight="bold")
    axBeforeBeamform.set_title("signal on the first element", fontweight="bold")
    axAfterBeamform.set_title("beamformer output", fontweight="bold")
    axExpectSpectrum.set_title("spectrum of expect signal", fontweight="bold")
    axInterferenceSpectrum.set_title("specturm of interference", fontweight="bold")
    axBeforeBeamformSpectrum.set_title("spectrum of signal on the first element", fontweight="bold")
    axAfterBeamformSpectrum.set_title("spectrum of output", fontweight="bold")
    axAmplitudeResponse.set_title("amplitude response", fontweight="bold")
    axPhaseResponse.set_title("phase response", fontweight="bold")

    spectrumX = fftfreq(fastSnapshots)

    # adjust axes
    axExpect.sharex(axInterference)
    axInterference.sharex(axBeforeBeamform)
    axBeforeBeamform.sharex(axAfterBeamform)
    axAfterBeamform.set_ylim(-1, 1)
    for ax in (axsForResponse):
        ax.set_xlim([-90, 90])
    axPhaseResponse.set_ylim([-np.pi * 1.2, np.pi * 1.2])
    axAmplitudeResponse.set_ylim([-70, 0])
    for ax in (axsForSpectrum):
        ax.set_xlim([spectrumX[0], spectrumX[-1]])
        ax.set_ylim([0, 1.2])

    # plot the expect signal and the interference signal and the received signal
    expectSignal.sample(samplePoints)
    coherentSignal.sample(samplePoints)
    axExpect.plot(np.real(expectSignal.signal), color=colors[0])
    axInterference.plot(np.real(coherentSignal.signal), color=colors[0])
    axBeforeBeamform.plot(np.real(output[0, :]), color=colors[0])

    # add vertical line to axes
    axExpectLeft = axExpect.axvline(1 - fastSnapshots, color=colors[1])
    axExpectRight = axExpect.axvline(0, color=colors[1])
    axInterferenceLeft = axInterference.axvline(1 - fastSnapshots, color=colors[1])
    axInterferenceRight = axInterference.axvline(0, color=colors[1])
    axBeforeBeamformLeft = axBeforeBeamform.axvline(1 - fastSnapshots, color=colors[1])
    axBeforeBeamformRight = axBeforeBeamform.axvline(0, color=colors[1])
    axAfterBeamformLeft = axAfterBeamform.axvline(1 - fastSnapshots, color=colors[1])
    axAfterBeamformRight = axAfterBeamform.axvline(0, color=colors[1])
    leftLines = (axExpectLeft, axInterferenceLeft, axBeforeBeamformLeft, axAfterBeamformLeft)
    rightLines = (axExpectRight, axInterferenceRight, axBeforeBeamformRight, axAfterBeamformRight)
    axAmplitudeResponse.axvline(expectTheta, color=colors[1])
    axAmplitudeResponse.axvline(coherentTheta, color=colors[2])
    axPhaseResponse.axvline(expectTheta, color=colors[1])
    axPhaseResponse.axvline(coherentTheta, color=colors[2])

    # lines for beamformer output, spectrum, response
    beamformOutputX, beamformOutputY = [], []
    lineForOutput, = axAfterBeamform.plot(beamformOutputX, beamformOutputY, color=colors[0])
    lineForExpectSpecturm, = axExpectSpectrum.plot([], [], color=colors[0])
    lineForInterferenceSpectrum, = axInterferenceSpectrum.plot([], [], color=colors[0])
    lineForBeforeSpectrum, = axBeforeBeamformSpectrum.plot([], [], color=colors[0])
    lineForAfterSpectrum, = axAfterBeamformSpectrum.plot([], [], color=colors[0])
    linesForSpectrum = (lineForExpectSpecturm, lineForInterferenceSpectrum, lineForBeforeSpectrum, lineForAfterSpectrum)
    lineForAmplitudeResponse, = axAmplitudeResponse.plot([], [], color=colors[0])
    lineForPhaseResponse, = axPhaseResponse.plot([], [], color=colors[0])
    linesForResponse = (lineForAmplitudeResponse, lineForPhaseResponse)

    def ani_func(num):
        for leftLine in leftLines:
            leftLine.set_xdata([num+1-fastSnapshots, num+1-fastSnapshots])
        for rightLine in rightLines:
            rightLine.set_xdata([num, num])

        # get the signal, expect, interference, array signal, output.
        if num < fastSnapshots - 1:
            usedOutput = np.concatenate((np.zeros((16, fastSnapshots-num-1), dtype=np.complex128), output[:, :num+1]), axis=1)
            usedExpect = np.concatenate((np.zeros(fastSnapshots-num-1, dtype=np.complex128), expectSignal.signal[:num+1]))
            usedInterference = np.concatenate((np.zeros(fastSnapshots-num-1, dtype=np.complex128), coherentSignal.signal[:num+1]))
            usedArraySignal = np.concatenate((np.zeros(fastSnapshots-num-1, dtype=np.complex128), output[0][:num+1]))
        else:
            usedOutput = output[:, num-fastSnapshots+1: num+1]
            usedExpect = expectSignal.signal[num-fastSnapshots+1: num+1]
            usedInterference = coherentSignal.signal[num-fastSnapshots+1: num+1]
            usedArraySignal = output[0][num-fastSnapshots+1: num+1]
        weight, beamformerOutput = mvdr(usedOutput, expectTheta, ary.steer_vector, returnOutput=True)  # change this line to use other method

        # change the output line
        toAppend = np.real(beamformerOutput[-1])
        beamformOutputX.append(num)
        beamformOutputY.append(toAppend)
        lineForOutput.set_xdata(beamformOutputX)
        lineForOutput.set_ydata(beamformOutputY)

        # change the spectrum
        for line in linesForSpectrum:
            line.set_xdata(spectrumX)
        lineForExpectSpecturm.set_ydata(normalize(np.abs(fft(usedExpect))))
        lineForInterferenceSpectrum.set_ydata(normalize(np.abs(fft(usedInterference))))
        lineForBeforeSpectrum.set_ydata(normalize(np.abs(fft(usedArraySignal))))
        lineForAfterSpectrum.set_ydata(normalize(np.abs(fft(beamformerOutput))))

        # change the response line
        aryForResponsePlot = aray_maker(weight.size, 0.5)
        response, thetas = aryForResponsePlot.response_with(-90, 90, plotThetaNumbers, weight, True)
        for line in linesForResponse:
            line.set_xdata(thetas)
        lineForPhaseResponse.set_ydata(np.angle(response))
        lineForAmplitudeResponse.set_ydata(value_to_decibel(np.abs(response), minRes=-50))

    ani = animation.FuncAnimation(figForAni, ani_func, frames=samplePoints, interval=2)
    ani.save('mvdr输入和输出.mp4', dpi=200, fps=60)

def data_generator():
    """
    data_generator for data generate function.
    """
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
    outputWithDifferentMethod()
    beamformAnimation()
    # generate_flag = input('generate data?y/n\n')
    # if generate_flag == 'y':
    #     data_generate(data_generator(), SAVE_PATH)
