__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def rev_exp(array, alpha):
    return np.exp(array)**(-1./alpha)

def plot_rev_exp(x_array, alpha, color):
    y_arr = rev_exp(x_array, alpha)
    plt.plot(x_array, y_arr, label="decay half-time: %s"%alpha, color=color)
    y_arr = y_arr*(1+np.random.normal(0, 0.05, y_arr.shape))
    y_arr /= y_arr[0]
    plt.plot(x_array, y_arr, 'o', color=color)


def model_mixture(separation, prevalence_of_1st):
    x_arr = np.linspace(-7, 7, 100)
    plt.title('%s std delta - %s%% - %s%%'% (separation, prevalence_of_1st*100, (1-prevalence_of_1st)*100))
    plt.plot(x_arr, prevalence_of_1st*norm.pdf(x_arr, loc=-separation/2.), 'r')
    plt.plot(x_arr, (1-prevalence_of_1st)*norm.pdf(x_arr, loc=separation/2.), 'b')
    plt.plot(x_arr, prevalence_of_1st*norm.pdf(x_arr, loc=-separation/2.)
             + (1-prevalence_of_1st)*norm.pdf(x_arr, loc=separation/2.), 'k')

if __name__ == "__main__":
    x_arr = np.linspace(0, 100, 100)
    # x_arr = np.array([0, 3, 6, 9, 12, 15, 23, 26, 32, 36, 39, 50, 60, 77, 100])
    # plt.title("Exponential decay with different decay half-times")
    # plot_rev_exp(x_arr, 0.25, 'black')
    # plot_rev_exp(x_arr, 7, 'green')
    # plot_rev_exp(x_arr, 20, 'red')
    # plot_rev_exp(x_arr, 100, 'magenta')
    # plt.legend()
    # plt.show()

    plt.subplot(331)
    model_mixture(8, 0.5)
    plt.subplot(332)
    model_mixture(4, 0.5)
    plt.subplot(333)
    model_mixture(2, 0.5)

    plt.subplot(334)
    model_mixture(2, 0.2)
    plt.subplot(335)
    model_mixture(2, 0.33)
    plt.subplot(336)
    model_mixture(2, 0.45)

    plt.subplot(337)
    model_mixture(2, 0.05)
    plt.subplot(338)
    model_mixture(4, 0.05)
    plt.subplot(339)
    model_mixture(8, 0.05)

    plt.show()