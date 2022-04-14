import numpy as np
import matplotlib.pyplot as plt
import signl
from utils import value_to_decibel

class Element:

    def __init__(self):
        self.place = None
        self.position = None

    def add_to(self, array, position):
        self.place = array
        self.position = position

    def remove_from(self, array):
        if self.place is array:
            self.place = None
            self.position = None

class ReferenceElement(Element):
    """
    参考阵元，基准阵元
    """

    def __init__(self):
        Element.__init__(self)
        self.signals = []

class Array:

    noise_power = 1
    implemented = False
    noise_reproducible = True  # for every element its noise is reproducible

    def __init__(self, size, reference_position):
        assert self.check_size(size)
        self.size = size
        assert self.check_position(reference_position), "参考位置错误"
        self.reference = ReferenceElement()
        self.reference.add_to(self, reference_position)
        self.elements = []
        # self.noise = signl.GaussionNoise()
        self.sample_points = None

    @property
    def output(self):
        assert self.sample_points is not None, '尚未采样，首先使用sample方法采样'
        res = []
        for element in self.elements:
            res.append(self.signal_at(element))
        return np.array(res)

    @property
    def signals(self):
        return self.reference.signals

    @property
    def element_number(self):
        return len(self.elements)

    def add_element(self, element, position):
        """
        通过该方法来添加阵元
        """
        assert isinstance(element, Element), '只能添加Element'
        assert self.check_position(position), '不合法的放置位置'
        assert element not in self.elements, '不能重复添加阵元'
        self.elements.append(element)
        element.add_to(self, position)

    def receive_signal(self, signal):
        """
        通过此方法入射信号
        """
        assert self.check_signal(signal), "{} 不是一个有效的信号".format(signal)
        self.reference.signals.append(signal)

    def sample(self, points):
        """
        对接收到的信号采样
        """
        assert isinstance(points, int) and points > 0, '信号点数错误'
        for signal in self.signals:
            signal.sample(points)
        self.sample_points = points

    def signal_at(self, element):
        assert element in self.elements, '该阵元不在阵列上'
        assert self.sample_points is not None, '需要先对信号采样，先应用sample方法'
        amplitude = np.sqrt(self.noise_power / 2)
        if self.noise_reproducible:
            index = self.elements.index(element)
            rng = np.random.default_rng(seed=index)
        else:
            rng = np.random.default_rng()
        noise = rng.standard_normal(self.sample_points) * amplitude + rng.standard_normal(self.sample_points) * amplitude * 1j
        return np.sum(self._signal_at(element), axis=0, keepdims=False) + noise

    def remove_element(self, element):
        assert element in self.elements, '{}不在阵元上'.format(element)
        self.elements.remove(element)
        element.remove_from(self)

    def remove_all_signal(self):
        self.reference.signals = []

    def steer_vector(self, theta):
        """
        导向矢量，列向量（n*1)
        """
        assert self.is_good_theta(theta), '角度错误'
        assert self.elements, '无阵元，无导向矢量'
        phase_diff_res = []
        for element in self.elements:
            phase_diff = self.phase_diff_with_theta(element, theta)
            phase_diff_res.append(phase_diff)
        phase_diff_res = np.array([[item] for item in phase_diff_res])
        return np.exp(1j * phase_diff_res)

    def phase_diff_with_theta(self, element, theta):
        return self.path_diff_with_theta(element, theta) * 2 * np.pi

    def response_at_with(self, theta, weight_vector):
        """
        在某个角度上的响应
        """
        assert self.elements, '无阵元，无响应'
        assert self.is_good_theta(theta), '错误的角度'
        assert self.check_weight_vector(weight_vector), '权矢量不合法'
        res = np.matmul(np.conjugate(weight_vector.T), self.steer_vector(theta))
        return res.item()

    def check_weight_vector(self, weight_vector):
        return isinstance(weight_vector, np.ndarray) and weight_vector.shape == (self.element_number, 1)

    def _signal_at(self, element):
        """
        某个阵元上所有的相移后的接收信号，如果没有接受信号返回空的数组
        """
        phase_shifted_signal = []
        for signal in self.reference.signals:
            phase_shifted_signal.append(
                    signal.signal *
                    np.exp(1j * self.phase_diff_with_theta(element, signal.theta))
                    )
        return np.array(phase_shifted_signal)

    def is_good_theta(self, theta):
        """
        需要重写
        """
        return True

    def is_good_points(self, points):
        """
        需要重写
        """
        return True

    def check_size(self, size):
        """
        需要重写
        """
        return True

    def check_position(self, position):
        """
        放置位置是否合法，需要重写
        """
        return True

    def check_signal(self, signal):
        """
        需要重写
        """
        if isinstance(signal, signl.Signal) and isinstance(signal, signl.Wave):
            if signal in self.reference.signals:
                return False
            elif self.reference.signals:
                return signal.signal_length == self.reference.signals[0].signal_length
            else:
                return True
        else:
            return False

    def path_diff_with_theta(self, element, theta):
        """
        需要重写
        """
        return 0

    def steer_vectors(self, begin, end, points, rettheta):
        """
        阵列流形，需要重写
        """
        assert self.is_good_theta(begin) and self.is_good_theta(end), '错误的角度'
        assert self.is_good_points(points), '错误的点数'
        assert self.elements, '无阵元，无阵列流形'

    def response_plot(self, weight_vector, fig_ax_pair=None, **plot_kwargs):
        """
        绘制响应曲线，需要重写
        """
        assert self.check_weight_vector(weight_vector)

    def response_with(self, begin, end, points, weight_vector, rettheta=False):
        """
        需要重写
        """
        assert self.elements, '无阵元，无响应'
        assert self.is_good_theta(begin) and self.is_good_theta(end), '错误的角度'
        assert self.is_good_points(points), '错误的点数'
        assert self.check_weight_vector(weight_vector), '权矢量不合法'

class LineArray(Array):

    implemented = True

    def __init__(self, length=float('inf'), reference_position=0):
        Array.__init__(self, length, reference_position)

    def check_size(self, size):
        return isinstance(size, (int, float)) and size > 0

    def check_position(self, position):
        return (Array.check_position(self, position)
                and isinstance(position, (int, float))
                and 0 <= position <= self.size
                )

    def check_signal(self, signal):
        return Array.check_signal(self, signal) and isinstance(signal, signl.Wave2D)

    def path_diff_with_theta(self, element, theta):
        distance = element.position - self.reference.position
        path_diff = distance * np.sin(np.deg2rad(theta))
        return path_diff

    def steer_vectors(self, begin, end, points, rettheta=False):
        Array.steer_vectors(self, begin, end, points, rettheta)
        thetas = np.linspace(begin, end, points, endpoint=True)
        res = []
        for theta in thetas:
            res.append(self.steer_vector(theta))
        if rettheta:
            return np.concatenate(res, axis=1), thetas
        else:
            return np.concatenate(res, axis=1)

    def response_with(self, begin, end, points, weight_vector, rettheta):
        Array.response_with(self, begin, end, points, weight_vector, rettheta)
        if rettheta:
            steer_vectors, thetas = self.steer_vectors(begin, end, points, rettheta=True)
        else:
            steer_vectors = self.steer_vectors(begin, end, points, rettheta=False)
        response = np.matmul(np.conjugate(weight_vector.T), steer_vectors)
        if rettheta:
            return response.flatten(), thetas
        else:
            return response.flatten()

    def response_plot(self, weight_vector, fig_ax_pair=(None, None), **plot_kwargs):
        Array.response_plot(self, weight_vector)
        fig, ax = fig_ax_pair
        if fig is None:
            fig, ax = plt.subplots()
        if self.elements:
            response, thetas = self.response_with(-90, 90, 3601, weight_vector, True)
            ax.plot(thetas, value_to_decibel(np.abs(response)), **plot_kwargs)
            fig.show()
        return fig, ax

    def is_good_theta(self, theta):
        return Array.is_good_theta(self, theta) and isinstance(theta, (int, float)) and -90 <= theta <= 90

    def is_good_points(self, points):
        return Array.is_good_points(self, points) and isinstance(points, int) and points > 0

class UniformLineArray(LineArray):

    implemented = True

    def __init__(self, length=float('inf'), reference_position=0, interval=0.5):
        """
        默认间隔为半波长
        """
        assert isinstance(interval, (int, float)) and interval > 0, '错误的阵元间距'
        LineArray.__init__(self, length, reference_position)
        self.interval = interval
        self.position_of_first = 0

    def add_element(self, element):
        new_position = 0
        if not self.elements:
            new_position = self.position_of_first
        else:
            new_position = self.elements[-1].position + self.interval
        LineArray.add_element(self, element, new_position)

    @staticmethod
    def with_settings(settings):
        keys_needed = [
                'signals',
                'element_number',
                'length',
                'reference_position',
                'interval',
                'sample_points'
                ]
        for key in keys_needed:
            assert key in settings, '设置中缺少{}'.format(key)
        newly_created = UniformLineArray(
                settings['length'],
                settings['reference_position'],
                settings['interval']
                )
        for k in range(settings['element_number']):
            newly_created.add_element(Element())
        for signal in settings['signals']:
            newly_created.receive_signal(signal)
        newly_created.sample(settings['sample_points'])
        return newly_created

class SurfaceArray(Array):
    implemented = False

if __name__ == "__main__":
    def my_plot(x):
        fig, ax = plt.subplots()
        ax.plot(x.real)
        fig.show()

    def do_wrong(x):
        try:
            exec(x)
        except AssertionError:
            pass
        else:
            assert False
    sample_points = 1000

    ary = Array(1, 2)
    ary.add_element(Element(), 1)
    ary.sample(sample_points)
    output1 = ary.output
    assert output1.shape == (1, sample_points)
    noise1 = np.matmul(output1, np.conjugate(output1.T)).item() / output1.shape[1]
    ary.receive_signal(signl.LfmWave2D())
    ary.sample(sample_points)
    output2 = ary.output
    assert output2.shape == (1, sample_points)
    noise2 = np.matmul(output2, np.conjugate(output2.T)).item() / output2.shape[1]
    print('noise1: {}, noise2: {} should be around 1'.format(noise1, noise2))

    ary = UniformLineArray()
    for k in range(8):
        ary.add_element(Element())
    signal1 = signl.LfmWave2D()
    signal1.amplitude = 10
    ary.receive_signal(signal1)
    ary.sample(sample_points)
    output1 = ary.output
    assert output1.shape == (8, sample_points)
    my_plot(output1[0])
    my_plot(output1[3])

    ary = Array(1, 1)
    assert len(ary.signals) == 0
    for k in range(2):
        ary.receive_signal(signl.LfmWave2D())
    assert len(ary.signals) == 2
    ary.remove_all_signal()
    assert len(ary.signals) == 0

    ary = LineArray()
    for k in range(8):
        ary.add_element(Element(), 10)
    assert ary.element_number == 8

    do_wrong('ary.add_element(Element(), -1)')
    do_wrong('ary.add_element(ary.elements[0], 1)')
    do_wrong('ary.add_element(1, 1)')

    ary = Array(1, 1)
    do_wrong('ary.receive_signal(1)')
    do_wrong('ary.receive_signal(signl.Lfm())')
    ary.receive_signal(signl.LfmWave2D())
    do_wrong('ary.receive_signal(ary.signals[0])')
    do_wrong('ary.receive_signal(signl.LfmWave2D(signal_length=10))')
    ary.receive_signal(signl.CosWave2D())
    assert len(ary.signals) == 2

    ary = UniformLineArray()
    for k in range(4):
        ary.add_element(Element())
    my_weigth = ary.steer_vector(0)
    assert ary.steer_vectors(2, 10, 8).shape == (4, 8)
    ary.response_plot(my_weigth)

    print('all test passed')
