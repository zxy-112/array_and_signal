from math import floor
import numpy as np
import matplotlib.pyplot as plt

def is_num(x):
    return isinstance(x, (int, float))

def is_positive(x):
    return is_num(x) and x > 0

class Signal:

    def __init__(self, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        assert is_positive(signal_length), '信号长度为正数值'
        assert is_positive(amplitude)
        for item in (fre_shift, phase_shift, delay):
            assert is_num(item)
        assert isinstance(signal_type, str)

        self.signal_length = signal_length
        self._signal = None
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.fre_shift = fre_shift
        self.delay = delay
        self.signal_type = signal_type

    def sample(self, points):
        """
        指定信号采样点数
        """
        assert isinstance(points, int) and points >=0, '采样点数不正确'
        if points == 0:
            self._signal = None
        else:
            t = np.linspace(0, self.signal_length, points, endpoint=False, retstep=False)
            self._signal = self.expression(t)
            if self.delay:
                delay_points = floor(self.delay * points / self.signal_length)
                if delay_points > 0:
                    self._signal[delay_points:], self._signal[:delay_points] = self._signal[:-delay_points], 0
                else:
                    self._signal[:delay_points], self.signal[delay_points:] = self._signal[-delay_points:], 0
        return self.signal

    def expression(self, t):
        """
        信号表达式，需要重写
        """
        return self.amplitude * np.exp(1j * 2 * np.pi * self.fre_shift * t) * np.exp(1j * self.phase_shift)

    def plot(self):
        """
        绘制信号波形，某些信号需要重写以使图形美观
        """
        fig, ax = None, None
        if self._signal is not None:
            fig, ax = plt.subplots()
            ax.plot(self._signal.real)
            ax.set_ylim((-1.2 * self.amplitude, 1.2 * self.amplitude))
            fig.show()
        return fig, ax

    @property
    def band_width(self):
        """
        信号带宽，需要重写
        """
        return 1 / self.signal_length

    @property
    def signal(self):
        """
        需要sample后才有信号
        """
        assert self._signal is not None, '需要先采样'
        if self._signal is not None:
            return self._signal
        else:
            raise NoSampleError('信号未采样')

    def __str__(self):
        return 'Signal with length {} at {}'.format(self.signal_length, id(self))

class Wave:

    def __init__(self, theta):
        self.theta = theta

class Wave2D(Wave):

    def __init__(self, theta=0):
        assert isinstance(theta, (int, float)) and -90 <= theta <= 90, '入射角不合法'
        Wave.__init__(self, theta)

class Wave3D(Wave):

    def __init__(self, theta=(0, 0)):
        assert isinstance(theta, (tuple, list)) and len(theta) == 2, '入射角不合法'
        Wave.__init__(self, theta)

class Lfm(Signal):

    def __init__(self, signal_length=10e-6, band_width=10e6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        assert isinstance(band_width, (int, float)) and band_width > 0, '信号带宽不合法'
        Signal.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        self._band_width = band_width

    def expression(self, t):
        mu = self._band_width / self.signal_length
        power_of_t = np.power(t - self.signal_length/2, 2)
        return Signal.expression(self, t) * np.exp(1j * np.pi * mu * power_of_t)

    @property
    def band_width(self):
        return self._band_width

    @band_width.setter
    def band_width(self, value):
        assert isinstance(value, (int, float)) and value > 0, '信号带宽不合法'
        self._band_width = value

class LfmWave2D(Lfm, Wave2D):
    """
    可以入射到线阵
    """

    def __init__(self, theta=0, signal_length=10e-6, band_width=10e6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Lfm.__init__(
                self,
                signal_length=signal_length,
                band_width=band_width,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta)

class LfmWave3D(Lfm, Wave3D):

    def __init__(self, theta=(0, 0), signal_length=10e-6, band_width=10e6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Lfm.__init__(
                self,
                signal_length=signal_length,
                band_width=band_width,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave3D.__init__(self, theta)

class Cos(Signal):

    pass

class CosWave2D(Cos, Wave2D):

    def __init__(self, theta=0, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Cos.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta)

class CosWave3D(Cos, Wave3D):

    def __init__(self, theta=(0, 0), signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Cos.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave3D.__init__(self, theta)

class GaussionNoise(Signal):

    def __init__(self, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Signal.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        self.points = None

    def sample(self, points):
        Signal.sample(self, points)
        self.points = points

    def expression(self, t):
        power = self.amplitude ** 2
        amplitude = np.sqrt(power / 2)
        return amplitude * (np.random.randn(*t.shape) + 1j * np.random.randn(*t.shape))

    def plot(self):
        fig, ax = Signal.plot(self)
        if ax is not None:
            max_value = np.max(np.abs(self.signal)) * 1.2
            ax.set_ylim([-max_value, max_value])
        return fig, ax

    @property
    def band_width(self):
        if self.points is None:
            return None
        else:
            return self.points / self.signal_length

class NoiseWave2D(GaussionNoise, Wave2D):

    def __init__(self, theta=0, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        GaussionNoise.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta)

class NoiseWave3D(GaussionNoise, Wave3D):

    def __init__(self, theta=(0, 0), signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        GaussionNoise.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave3D.__init__(self, theta)

class NoSampleError(BaseException):
    pass

if __name__ == '__main__':
    def do_wrong(string):
        print(dir())
        try:
            exec(string)
        except AssertionError:
            pass
        else:
            assert False
    sample_points = 1000
    do_wrong('signal = Signal(-10)')
    signal = Signal(10e-6)
    print(signal)
    do_wrong('signal.signal')
    signal.sample(sample_points)
    assert signal.signal.shape == (sample_points,)
    print('signal_length: {}'.format(signal.signal_length))
    print('signal bandwidth: {}'.format(signal.band_width))
    signal.plot()
    signal.fre_shift = 1e6
    signal.sample(sample_points)
    signal.plot()
    signal.phase_shift = 0.5 * np.pi
    signal.sample(sample_points)
    signal.plot()
    lfm = LfmWave2D()
    do_wrong('lfm.signal')
    lfm.plot()  # 无事发生
    lfm.sample(sample_points)
    lfm.plot()
    print('lfm bandwidth: {}'.format(lfm.band_width))
    lfm.band_width = 5e6
    lfm.sample(sample_points)
    lfm.plot()
    do_wrong('lfm = LfmWave2D(signal_length=-1)')
    noise = GaussionNoise(10e-6)
    print('noise bandwidth: {}'.format(noise.band_width))
    noise.sample(sample_points)
    print('noise bandwidth: {}'.format(noise.band_width))

    print('all test passed')
