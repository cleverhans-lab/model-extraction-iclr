from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.stats as st
import math
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt


class bounded_noise(st.rv_continuous):
    def _pdf(self, x, bound, normalize):
        return np.exp(-np.exp(1 / (1 - (x / bound) ** 2))) / normalize


class bounded_noise_2(st.rv_continuous):
    def _pdf(self, x):
        return np.exp(-np.exp(1 / (1 - (x) ** 2)))

def sample_gaussian_noise(epsilon, delta, num_labels, universal_constant, shape,
                         num_users):
    # We need to determine how to set universal_constant
    sigma = universal_constant * np.sqrt(num_labels * np.log(1 / delta)) / (
                epsilon * (float(num_users)))
    assert sigma > 0
    assert sigma != np.inf
    assert sigma != -np.inf
    return sigma * np.random.randn(shape), sigma

def sample_laplace_noise(epsilon, delta, num_labels, universal_constant, shape,
                         num_users):
    # We need to determine how to set universal_constant
    sigma = universal_constant * np.sqrt(num_labels * np.log(1 / delta)) / (
                epsilon * (float(num_users)))
    assert sigma > 0
    assert sigma != np.inf
    assert sigma != -np.inf
    return np.random.laplace(loc=0.0, scale=1./epsilon, size=shape).reshape(shape), 1./epsilon

def sample_bounded_noise(epsilon, delta, num_labels, universal_constant, shape,
                         num_users, noise_type):
    # print(np.sqrt(num_labels*np.log(1/delta)))
    # print(epsilon)
    # print(shape[0])
    # print((epsilon*(float(shape[0]))))
    bound = universal_constant * np.sqrt(num_labels * np.log(1 / delta)) / (
            epsilon * (float(num_users)))
    assert bound > 0
    assert bound != np.inf
    assert bound != -np.inf
    # print("Bound")
    # print(bound)

    if noise_type == 'bounded':
        y = lambda x: np.exp(-np.exp(1 / (1 - (x / bound) ** 2)))
        normalize = integrate.quad(y, -bound, bound)[0]
        # print("Normalize")
        # print(normalize)

        distrib = bounded_noise(momtype=0, a=-bound, b=bound,
                                shapes='bound, normalize', xtol=1e-14,
                                name='bounded_noise')
        # distrib = bounded_noise_2(a= -bound, b = bound, badvalue=[1.0,-1.0] xtol=1e-14, name='bounded_noise_2')
        noise_matrix = distrib.rvs(size=shape, bound=bound, normalize=normalize)
    elif noise_type == 'gaussian':
        sigma = bound
        noise_matrix = sigma * np.random.randn(*shape)
    return noise_matrix, bound


class Multilabel_Accountant():
    def __init__(self, epsilon, delta, num_labels, universal_constant):
        assert universal_constant > 0
        self.epsilon = epsilon
        # print(np.exp(-num_labels / np.log(num_labels)**8))
        # assert delta >= np.exp(-num_labels / np.log(num_labels)**8)
        self.delta = delta
        self.num_labels = num_labels
        self.universal_constant = universal_constant

    def get_noisy_counts(self, prediction_matrix):
        num_samples = prediction_matrix.shape[0]
        # noise_matrix = self.sample_noise(prediction_matrix.shape)
        noise_matrix, _ = sample_bounded_noise(
            epsilon=self.epsilon,
            delta=self.delta,
            num_labels=self.num_labels,
            universal_constant=self.universal_constant)
        return prediction_matrix + noise_matrix


# random_pred_matrix = np.random.rand(100, 14)
# accountant = Multilabel_Accountant(0.01, 1.0 / (100 ** 2), 14, 5)
# accountant.get_noisy_counts(random_pred_matrix)
#
# samples = accountant.distrib.rvs(bound=accountant.bound, size=1000)
#
# # plot histogram of samples
# fig, ax1 = plt.subplots()
# # ax1.hist(list(samples), bins=50)
#
# # plot PDF and CDF of distribution
# pts = np.linspace(-accountant.bound, accountant.bound)
# ax2 = ax1.twinx()
# ax2.set_ylim(0, 0.2)
# print(pts)
# print(accountant.distrib.pdf(pts, bound=accountant.bound))
# ax2.plot(pts, accountant.distrib.pdf(pts, bound=accountant.bound), color='red')
#
# fig.tight_layout()
# plt.show()
# plt.savefig("testing.png")
