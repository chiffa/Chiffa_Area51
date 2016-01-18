"""
Contains MA plot and quintile normalization routine
"""
import numpy as np
from matplotlib import pyplot as plt
from general_plotting import show_with_labels

def quintile_normalize(array_1d_1, array_1d_2):
    """
    Performs quintile normalization on two arrays representing two condition of experiment on the same variables

    :param array_1d_1:
    :param array_1d_2:
    :return:
    """
    array_1d_1 -= np.nanmean(array_1d_1)
    array_1d_2 -= np.nanmean(array_1d_2)
    combined_distro = np.vstack((array_1d_1, array_1d_2))
    arg_sort_combined = np.argsort(combined_distro, axis=1)
    sort_combined = np.sort(combined_distro, axis=1)
    substitution_values = np.mean(sort_combined, axis=0)

    array_1d_1, array_1d_2 = np.vsplit(substitution_values[arg_sort_combined], 2)
    array_1d_1, array_1d_2 = (array_1d_1[0, :], array_1d_2[0, :])

    return  array_1d_1, array_1d_2


def ma_plot(data_1, data_2, variable_names,
            quintile_normalized=True, outlier_selector=None,
            experiment_1_name='condition 1', experiment_2_name='condition 2'):
    """
    MA plot

    :param data_1: have to be log10
    :param data_2: have to be log 10
    :param variable_names:
    :param quintile_normalized:
    :param outlier_selector:
    :param experiment_1_name:
    :param experiment_2_name:
    :return:
    """
    if quintile_normalized:
        data_1, data_2 = quintile_normalize(data_1, data_2)

    a_ = 0.5*(data_1 + data_2)
    m_ = data_1 - data_2

    plt.title('MA plot')
    show_with_labels(a_, m_, variable_names)
    plt.ylabel('%s - %s log10 ppm' % (experiment_1_name, experiment_2_name))
    if outlier_selector:
        plt.axhline(outlier_selector, color='red', lw=2)
    plt.legend()
    plt.show()

    if outlier_selector:
        if outlier_selector > 0:
            frac = m_ > outlier_selector
        elif outlier_selector < 0:
            frac = m_ < outlier_selector
        else:
            frac = np.logical_and(m_<0.1, m_>-0.1)

        return variable_names[frac]

    return []