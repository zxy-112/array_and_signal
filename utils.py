import numpy as np
import matplotlib.pyplot as plt

pinv = np.linalg.pinv
pinv_fast = lambda x: pinv(x, hermitian=True)

def value_to_decibel(vector):
    res = 20 * np.log10(vector / np.max(vector))
    res[res < -60] = -60
    return res

def make_dic_maker(dic):
    def maker(*funcs):
        new_dic = dic.copy()
        for func in funcs:
            func(new_dic)
        return new_dic
    return maker

def change_maker(key, value):
    def change(dic):
        dic[key] = value
    return change

def hermitian(matrix):
    return np.conjugate(matrix.T)

def calcu_cov(output):
    return np.matmul(output, hermitian(output)) / len(output)

def mvdr(output, steer_vector):
    inv_mat = pinv(calcu_cov(output))
    temp = np.matmul(inv_mat, steer_vector)
    return temp / (hermitian(steer_vector) @ inv_mat @ steer_vector)

def my_plot(*args, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(*args, **kwargs)
    fig.show()
    return fig, ax

def syn(weight, output):
    return np.matmul(hermitian(weight), output).squeeze()

