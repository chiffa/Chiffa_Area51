__author__ = 'Andrei'

import numpy as np
import os
from csv import reader
from matplotlib import pyplot as plt

work_folder = 'C:\\Users\\Andrei\\Desktop\\for_Linhao'
reference = '4932-WHOLE_ORGANISM-integrated.txt'
experiment = 'HS_6.txt'
experiments = ['HS_6', 'HS_15', 'H2O2_40', 'H2O2_70']
header_selection = ['0N-++--', '00N++---', '0N++--', '0N++--' ]

experiments_path = [os.path.join(work_folder, exp + '.txt') for exp in experiments]

reference = os.path.join(work_folder, reference)
experiment = os.path.join(work_folder, experiment)


def show_with_labels(x_arg, y_arg, label_arg):
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        x_arg, y_arg, marker='o'
    )
    for label, x, y in zip(label_arg, x_arg, y_arg):
        plt.annotate(
            label,
            xy=(x, y),
            fontsize=8)

    plt.show()


def interpolation_function(x):
    if x == 'X':
        return np.nan
    else:
        return np.float(x)


def vector_interpolation(x_list):
    return [interpolation_function(x) for x in x_list]


with open(reference, 'r') as ref_f:
    ref_iter = reader(ref_f, delimiter = '\t')
    ref_headers = ref_iter.next()
    ref_list = [line for line in ref_iter]
    ref_names = np.array([line[1].split('.')[1] for line in ref_list]).astype(np.str)
    ref_names = ref_names.flatten()
    ref_values = np.array([[float(line[2]), 0] for line in ref_list]).astype(np.float64)
    sorted_ref_values = ref_values[np.argsort(ref_names), 0][:, np.newaxis]


def lgi(lst, index_list):
    """
    List get indexes: recovers indexes in the list in the provided index list and returns the result in the form of an
    array
    :param lst:
    :param index_list:
    :return:
    """
    if index_list == []:
        return []
    if isinstance(index_list[0], int):
        return np.array([lst[i_] for i_ in index_list])
    if isinstance(index_list[0], bool) or isinstance(index_list[0], np.bool_):
        index_list = np.array(index_list).nonzero()[0]
        return np.array([lst[i_] for i_ in index_list])
    else:
        raise Exception('argument not understood')



def read_experiment(experiment_file, header_selection):
    msk = list(header_selection)
    name_idx = msk.index('N')
    msk = np.array(msk)
    positive = msk == '+'
    negative = msk == '-'

    with open(experiment_file, 'r') as exp_f:
        exp_iter = reader(exp_f, delimiter = '\t')
        exp_iter.next()
        exp_list = [line for line in exp_iter]
        exp_names = np.array([line[name_idx].split('\.') for line in exp_list]).astype(np.str)
        exp_names = exp_names.flatten()
        exp_values_pos = np.nanmean(np.array([vector_interpolation(lgi(line, positive)) for line in exp_list ]).astype(np.float64), axis=1)
        exp_values_neg = np.nanmean(np.array([vector_interpolation(lgi(line, negative)) for line in exp_list ]).astype(np.float64), axis=1)

    set_to_add = np.array(list(set(ref_names.tolist()) - set(exp_names.tolist())))
    post_exp_names = np.hstack((exp_names, set_to_add))
    argsorter = np.argsort(post_exp_names)
    value_data = argsorter[:len(exp_names)]

    # yes, we did inject ref_values wrongly around here
    ret_pair = np.empty_like(ref_values)
    ret_pair.fill(np.nan)

    ret_pair[value_data, 0] = exp_values_pos
    ret_pair[value_data, 1] = exp_values_neg
    ret_pair[ret_pair == 0] = np.nan

    return ret_pair


def read_all_experiments():
    exp_accumulator = [sorted_ref_values]

    for exp, selector in zip(experiments_path, header_selection):
        print exp, selector
        exp_accumulator.append(read_experiment(exp, selector))

    exp_accumulator = np.hstack(exp_accumulator)
    print exp_accumulator.shape
    return np.hstack((np.sort(ref_names)[:, np.newaxis], exp_accumulator))


super_table = read_all_experiments()

active = super_table[:, 2::2].astype(np.float64)
negative_control = super_table[:, 3::2].astype(np.float64)

negative_indexes = np.logical_not(np.all(np.isnan(negative_control), axis=1))
active_indexes = np.logical_and(np.logical_not(np.all(np.isnan(active), axis=1)), np.logical_not(negative_indexes))
active_names = super_table[active_indexes, 0]
active_2 = active[active_indexes, :]
negative_names = super_table[negative_indexes, 0]
negative_control_2 = negative_control[negative_indexes, :]

print active_2.shape, negative_control_2.shape

new_active_index = np.logical_and(active_indexes, np.logical_not(np.any(np.isnan(active), axis=1)))

# for i, exp_name in enumerate(experiments):
#     plt.loglog(sorted_ref_values[active_indexes], active_2[:, i], 'o', label=exp_name)

# plt.loglog(sorted_ref_values[new_active_index], np.nanmean(active[new_active_index, 1:2], axis=1), 'ko', label="common to HS")
show_with_labels(np.log(sorted_ref_values[new_active_index]),
                 np.log(np.nanmean(active[new_active_index, :], axis=1)),
                 ref_names[new_active_index])
# TODO: TRANSLATE into GENE names
plt.legend()
plt.show()

# TODO : pull all experiments into a single table
# TODO : do a DBScan to get outliers.
# TODO : suppress results that are mos likely due to noise => correlation plot no label v.s. label
