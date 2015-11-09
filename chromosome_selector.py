__author__ = 'Andrei'

import os
import numpy as np
from csv import reader, writer


work_folder = 'C:\\Users\\Andrei\\Desktop'
input = 'yeast_gene_locations.txt'
output = 'Chr_10.txt'

input = os.path.join(work_folder, input)
output = os.path.join(work_folder, output)

with open(input, 'r') as ref_f:
    ref_iter = reader(ref_f, delimiter = '\t')
    ref_headers = ref_iter.next()
    ref_list = [line for line in ref_iter]
    ref_names = np.array([line[2] for line in ref_list]).astype(np.str)
    ref_names = ref_names.flatten()
    ref_values = np.array([line[3] for line in ref_list]).astype(np.str)

Chr_X = ref_names[ref_values == 'X']
Chr_X = Chr_X[Chr_X != '']

with open(output, 'w') as out_f:
    out_writer = writer(out_f, delimiter = '\t', lineterminator = '\n')
    out_writer.writerows(Chr_X[:, np.newaxis].tolist())
