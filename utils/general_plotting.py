from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def show_with_labels(x_arg, y_arg, label_arg):
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        x_arg, y_arg, marker='o'
    )
    print len(label_arg), x_arg.shape, y_arg.shape
    for label, x, y in zip(label_arg, x_arg, y_arg):
        plt.annotate(
            label,
            xy=(x, y),
            fontsize=8)

    # plt.show()


def smooth_histogram(data, color='k', bins = 200):
    fltr = np.logical_not(np.isnan(data))
    density = gaussian_kde(data[fltr].flatten())
    xs = np.linspace(data[fltr].min(), data[fltr].max(), bins)
    plt.plot(xs, density(xs), color)