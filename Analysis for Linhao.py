__author__ = 'Andrei'

import numpy as np
import os
from csv import reader
from matplotlib import pyplot as plt
from utils.general_plotting import show_with_labels
from utils.list_manipulation import list_get_indexes
from utils.ma_plot import ma_plot

work_folder = 'C:\\Users\\Andrei\\Desktop\\for_Linhao'
reference = '4932-WHOLE_ORGANISM-integrated.txt'
experiment = 'HS_6.txt'
experiments = ['HS_6', 'HS_30', 'H2O2_40', 'H2O2_70']
header_selection = ['HN-++--', '0HN++----', 'HN++--', 'HN++--' ]

experiments_path = [os.path.join(work_folder, exp + '.txt') for exp in experiments]

reference = os.path.join(work_folder, reference)
experiment = os.path.join(work_folder, experiment)


def parsing_function(x):
    if x == 'X':
        return np.nan
    else:
        return np.float(x)


def vector_interpolation(x_list):
    return [parsing_function(x) for x in x_list]


with open(reference, 'r') as ref_f:
    ref_iter = reader(ref_f, delimiter = '\t')
    ref_headers = ref_iter.next()
    ref_list = [line for line in ref_iter]
    ref_names = np.array([line[1].split('.')[1] for line in ref_list]).astype(np.str)
    ref_names = ref_names.flatten()
    ref_alt_names = np.empty_like(ref_names)
    ref_alt_names.fill(np.nan)
    ref_values = np.array([[float(line[2]), 0] for line in ref_list]).astype(np.float64)
    sorted_ref_values = ref_values[np.argsort(ref_names), 0][:, np.newaxis]
    name_translation_list = ref_names.copy()


def read_experiment(experiment_file, header_selection):
    msk = list(header_selection)
    name_idx = msk.index('N')
    hgn_idx = msk.index('H')
    msk = np.array(msk)
    positive = msk == '+'
    negative = msk == '-'

    with open(experiment_file, 'r') as exp_f:
        exp_iter = reader(exp_f, delimiter = '\t')
        exp_iter.next()
        exp_list = [line for line in exp_iter]
        exp_names = np.array([line[name_idx].split('\.') for line in exp_list]).astype(np.str)
        exp_names = exp_names.flatten()
        hgn_names = np.array([line[hgn_idx].split('\.') for line in exp_list]).astype(np.str).flatten()
        supressor = exp_names != 'YNL054W-B'
        exp_names = exp_names[supressor]
        hgn_names = hgn_names[supressor]
        exp_pos_comps = np.array([vector_interpolation(list_get_indexes(line, positive)) for line in exp_list]).astype(np.float64)
        exp_pos_comps = exp_pos_comps[supressor, :]
        exp_values_pos = np.nanmean(exp_pos_comps, axis=1)
        exp_pos_comp1 = exp_pos_comps[:, 0]
        exp_pos_comp2 = exp_pos_comps[:, 1]
        exp_neg_comps = np.array([vector_interpolation(list_get_indexes(_line, negative)) for _line in exp_list]).astype(np.float64)
        exp_neg_comps = exp_neg_comps[supressor, :]
        control_positive = np.logical_and(np.logical_not(np.isnan(exp_neg_comps)), exp_neg_comps != 0).astype(np.int8)
        exp_neg_once = np.logical_and(np.sum(control_positive, axis=1) > 0, np.sum(control_positive, axis=1) < 2)
        exp_neg_twiceplus = np.sum(control_positive, axis=1) > 1
        exp_values_neg = np.nanmean(exp_neg_comps, axis=1)

    # plt.title(experiment_file.split('\\')[-1].split('.')[0])
    # plt.loglog(exp_pos_comp1, exp_pos_comp2, 'ko', label = 'no negative')
    # plt.loglog(exp_pos_comp1[exp_neg_once], exp_pos_comp2[exp_neg_once], 'bo', label = 'negative once')
    # plt.loglog(exp_pos_comp1[exp_neg_twiceplus], exp_pos_comp2[exp_neg_twiceplus], 'go', label = 'negative_more_than_once')
    # plt.legend()
    # plt.show()

    set_to_add = np.array(list(set(ref_names.tolist()) - set(exp_names.tolist())))
    print 'illegal names \t: %s' % np.array(list(set(exp_names.tolist()) - set(ref_names.tolist())))
    print 'called/appended/total: \t %s/%s/%s' % (len(exp_names.tolist()), len(set_to_add),  len(exp_names.tolist()) + len(set_to_add))
    post_exp_names = np.hstack((exp_names, set_to_add))
    value_data = np.argsort(np.argsort(post_exp_names) )[:len(exp_names)] # obtain the arguments of position of the first X elements to insert them
    print 'order-matched: \t %s' % np.all(np.sort(post_exp_names)[value_data] == exp_names)

    ret_pair = np.empty_like(ref_values)
    ret_pair.fill(np.nan)

    ret_pair[value_data, 0] = exp_values_pos
    ret_pair[value_data, 1] = exp_values_neg
    ret_pair[ret_pair == 0] = np.nan
    hgn_names[hgn_names == ''] = name_translation_list[value_data][hgn_names == '']
    name_translation_list[value_data] = hgn_names

    return ret_pair


def read_all_experiments():
    exp_accumulator = [sorted_ref_values]

    for exp, selector in zip(experiments_path, header_selection):
        print 'loading %s with selector: %s' % (exp, selector)
        exp_accumulator.append(read_experiment(exp, selector))

    exp_accumulator = np.hstack(exp_accumulator)
    return np.hstack((np.sort(ref_names)[:, np.newaxis], exp_accumulator))


def render_selected_experiments(selector_1, selector_2, names=True, diag_select=None):

    name_selector = ['cell average'] + experiments
    new_active_index = np.logical_and(
            active_indexes, np.logical_not(np.any(np.isnan(experimental_data[:, [selector_1, selector_2]]), axis=1)))
    # new_active_index = np.logical_and(new_active_index, sorted_ref_values[:, 0]>5)
    # new_active_index = np.logical_and(new_active_index, active[:, 1]>5e-4)

    active_data1 = np.log10(experimental_data[new_active_index, selector_1])
    active_data2 = np.log10(experimental_data[new_active_index, selector_2])
    active_names = name_translation_list[new_active_index]
    diag_select = np.log10(diag_select)

    show_with_labels(active_data1, active_data2, active_names)

    plt.xlabel('log10 of %s ppm' % name_selector[selector_1])
    plt.ylabel('log10 of %s ppm' % name_selector[selector_2])
    plt.legend()
    plt.show()

    # MA PLOT
    ma_results = ma_plot(active_data1, active_data2, active_names,
                         quintile_normalized=True, outlier_selector=diag_select,
                         experiment_1_name=name_selector[selector_1], experiment_2_name=name_selector[selector_2])

    print ma_results


if __name__ == "__main__":

    super_table = read_all_experiments()  # reads all the experiments, but the final table is of type "string"

    reference_data = super_table[:, 1][:, np.newaxis].astype(np.float64)
    experimental_data = super_table[:, 2::2].astype(np.float64)  # extraction of the experimental data and conversion to float
    experimental_data = np.hstack((reference_data, experimental_data))
    negative_control = super_table[:, 3::2].astype(np.float64)  # extraction of negative control and conversion to float

    negative_indexes = np.sum(np.isnan(negative_control).astype(np.int8), axis=1) < 1  # all the indexes that fail the negative control test

    # we then select everyone who doesn't fail the test and who has been detected in at least one experiment
    active_indexes = np.logical_and(np.logical_not(np.all(np.isnan(experimental_data), axis=1)), np.logical_not(negative_indexes))

    # # we check that we have pulled the right elements until now:
    # print name_translation_list[active_indexes]

    active_names = super_table[active_indexes, 0]
    active_2 = experimental_data[active_indexes, :]
    # negative_names = super_table[negative_indexes, 0]
    # negative_control_2 = negative_control[negative_indexes, :]

    render_selected_experiments(0, 2, diag_select=0.1)

    # TODO: calculate the distribution of erros within the experiments
    #       use them to perform a DBScan
    #       use the outliers that are specific to TS_30 in order to perform the enrichment analysis
    #       use the outliers that are identical between TS_30 and TS_60 to perform enrichment analysis
    #       use the outliers that are not to perform enrichment analysis
    # TODO : do a DBScan to get outliers.

