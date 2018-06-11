from csv import reader as csv_read
import numpy as np
from utils.general_plotting import smooth_histogram
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from utils.linalg import moving_average



master_table = []
with open(my_file, 'r') as source:
    reader = csv_read(source)
    header = reader.next()
    for line in reader:
        line = [float(elt) if elt else np.nan for elt in line[1:-3]]
        master_table.append(line)


master_table = np.array(master_table)

D_r = master_table[master_table[:, -1] > 0., 1]
D_nr = master_table[master_table[:, -1] == 0., 1]

GM_r = master_table[master_table[:, -1] > 0., 0]
GM_nr = master_table[master_table[:, -1] == 0., 0]

S_r = master_table[master_table[:, -1] > 0., 2]
S_nr = master_table[master_table[:, -1] == 0., 2]

s_r = master_table[master_table[:, -1] > 0., -2]
s_nr = master_table[master_table[:, -1] == 0., -2]

s_3d_av = moving_average(master_table[np.logical_not(np.isnan(master_table[:, -1])), -2], 3)

plt.plot(master_table[:, -2], 'k')
plt.plot(s_3d_av, 'r')

plt.show()

print s_3d_av.shape

print s_3d_av

print ks_2samp(D_r, D_nr)
smooth_histogram(D_r, 'r', 10)
smooth_histogram(D_nr, 'k', 10)
plt.show()

print ks_2samp(GM_r, GM_nr)
smooth_histogram(GM_r, 'r', 10)
smooth_histogram(GM_nr, 'k', 10)
plt.show()

print ks_2samp(S_r, S_nr)
smooth_histogram(S_r, 'r', 10)
smooth_histogram(S_nr, 'k', 10)
plt.show()

print ks_2samp(s_r, s_nr)
smooth_histogram(s_r, 'r', 10)
smooth_histogram(s_nr, 'k', 10)
plt.show()