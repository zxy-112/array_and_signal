import numpy as np
import matplotlib.pyplot as plt

pinv = np.linalg.pinv
pinv_fast = lambda x: pinv(x, hermitian=True)

def value_to_decibel(vector):
    res = 20 * np.log10(vector / np.max(vector))
    res[res < -70] = -70
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

def mvdr(output, expect_theta, steer_func):
    inv_mat = pinv(calcu_cov(output))
    steer_vector = steer_func(expect_theta)
    temp = np.matmul(inv_mat, steer_vector)
    return temp / (hermitian(steer_vector) @ inv_mat @ steer_vector).item()

def my_plot(*args, fig_ax_pair=(None, None), **kwargs):
    if fig_ax_pair == (None, None):
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax_pair
    ax.plot(*args, **kwargs)
    fig.show()
    return fig, ax

def syn(weight, output):
    return np.matmul(hermitian(weight), output).squeeze()

f_then_g = lambda f, g: lambda x: f(g(x))
def funcs_cons(*funcs):
    identity = lambda x: x
    res_func = identity
    for func in funcs:
        res_func = f_then_g(res_func, func)
    return res_func

def cal_ac(expect_theta, coherent_theta, steer_func):
    matrix_ac = [steer_func(expect_theta)]
    if isinstance(coherent_theta, (tuple, list)):
        for theta in coherent_theta:
            matrix_ac.append(steer_func(theta))
    else:
        matrix_ac.append(steer_func(coherent_theta))
    matrix_ac = np.concatenate(matrix_ac, axis=1)
    return matrix_ac

def cal_bc(coherent_theta, steer_func):
    matrix_bc = cal_ac(0, coherent_theta, steer_func)
    matrix_bc[:, 0] = 0
    return matrix_bc

def cal_f_vec(coherent_theta):
    f_vec = [1]
    if isinstance(coherent_theta, (tuple, list)):
        for _ in coherent_theta:
            f_vec.append(0)
    else:
        f_vec.append(0)
    return np.array([[item] for item in f_vec])

def mcmv(output, expect_theta, coherent_theta, steer_func):
    inv_cov = f_then_g(pinv_fast, calcu_cov)(output)
    matrix_ac = cal_ac(expect_theta, coherent_theta, steer_func)
    vector_f = cal_f_vec(coherent_theta)
    mcmv_weight = (
            inv_cov @
            matrix_ac @
            pinv(hermitian(matrix_ac) @ inv_cov @ matrix_ac) @
            vector_f
            )
    return mcmv_weight

def ctmv(output, expect_theta, coherent_theta, steer_func, sigma_power, diagonal_load=0):
    cov_mat = calcu_cov(output)
    cov_mat_loaded = cov_mat + diagonal_load * np.eye(len(cov_mat))
    inv_cov = pinv_fast(cov_mat)
    inv_loaded_cov = pinv_fast(cov_mat_loaded)
    matrix_ac = cal_ac(expect_theta, coherent_theta, steer_func)
    matrix_bc = cal_bc(coherent_theta, steer_func)
    matrix_t = (
            np.eye(len(cov_mat)) -
            (matrix_ac - matrix_bc) @
            pinv_fast(hermitian(matrix_ac) @ inv_loaded_cov @ matrix_ac) @
            hermitian(matrix_ac) @
            inv_loaded_cov
            )
    newly_created_cov = (
            matrix_t @ cov_mat @ hermitian(matrix_t) -
            sigma_power * matrix_t @ hermitian(matrix_t) +
            sigma_power * np.eye(len(cov_mat))
            )
    steer_vector = steer_func(expect_theta)
    inv_newly_created_cov = pinv_fast(newly_created_cov)
    weight = (
            inv_newly_created_cov @ steer_vector /
            (hermitian(steer_vector) @ inv_newly_created_cov @ steer_vector)
            )
    return weight

def ctp(output, expect_theta, coherent_theta, steer_func, sigma_power):
    cov_mat = calcu_cov(output)
    inv_cov = pinv_fast(cov_mat)
    matrix_ac = cal_ac(expect_theta, coherent_theta, steer_func)
    matrix_t = (
            np.eye(len(cov_mat)) -
            matrix_ac @
            pinv_fast(hermitian(matrix_ac) @ inv_cov @ matrix_ac) @
            hermitian(matrix_ac) @
            inv_cov
            )
    newly_created_cov = (
            matrix_t @ cov_mat @ hermitian(matrix_t) -
            sigma_power * matrix_t @ hermitian(matrix_t) +
            sigma_power * np.eye(len(cov_mat))
            )
    u, s, _ = np.linalg.svd(newly_created_cov, hermitian=True)
    count = 0
    refer_sum = np.sum(s) / len(s)
    for item in s:
        if item > refer_sum:
            count += 1
    matrix_u = None
    if count == 0:
        matrix_u = np.zeros((len(cov_mat), 1))
    else:
        matrix_u = u[:, :count]
    u2, _, _ = np.linalg.svd(cov_mat, hermitian=True)
    # eig_vec = u2[:, :1]
    max_index = max(range(count+1), key=lambda index: np.linalg.norm((np.eye(len(cov_mat)) - matrix_t @ hermitian(matrix_t)) @ u2[:, index: index+1]))
    return (np.eye(len(cov_mat)) - matrix_t @ hermitian(matrix_t)) @ u2[:, max_index: max_index+1]

def duvall(output, expect_theta, steer_func):
    steer_vector = steer_func(expect_theta)
    output = np.multiply(output, np.conjugate(steer_vector))
    delta = output[:-1, :] - output[1:, :]
    return mvdr(delta, 0, lambda x: steer_func(x)[:-1, :])

