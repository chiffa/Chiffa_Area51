from csv import reader as csv_reader
import numpy as np
from matplotlib import pyplot as plt
import os

prefix = 'C:\Users\Andrei\Dropbox\workspaces\JHU\Ewald Lab\TF 2 targets'
loc_1 = os.path.join(prefix, 'human_cellnet_grn_Apr_05_2017.csv')
loc_2 = os.path.join(prefix, 'TRRUST\\trrust_rawdata.human.tsv')

compendium_1 = []
compendium_1_support = []
compendium_2 = []
compendium_2_support = []

with open(loc_1, 'rb') as source:
    reader = csv_reader(source, delimiter=',')
    header = reader.next()
    print header
    for line in reader:
        interaction_no = int(line[0])
        interaction_from = str(line[1])
        interaction_to = str(line[2])
        interaction_z_score = float(line[3])
        interaction_correlation = float(line[4])
        compendium_1.append((interaction_from, interaction_to))
        compendium_1_support.append((interaction_z_score, interaction_correlation))

with open(loc_2, 'rb') as source:
    reader = csv_reader(source, delimiter='\t')
    for line in reader:
        interaction_from = str(line[0])
        interaction_to = str(line[1])
        interaction_type = line[2]
        evidence = line[3].split(';')
        evidence_redundancy = len(evidence)
        compendium_2.append((interaction_from, interaction_to))
        compendium_2_support.append((evidence_redundancy))

intersection = set(compendium_1).intersection(set(compendium_2))

print len(compendium_1), len(compendium_2)
print 'intersection: ', len(intersection)

compendium_1 = np.array(compendium_1)
compendium_2 = np.array(compendium_2)

s1 = set(compendium_1.flatten().tolist())
s2 = set(compendium_2.flatten().tolist())
intersection2 = s1.intersection(s2)

print len(s1), len(s2)
print 'nodes intersection: ', len(intersection2)

compendium_1_support = np.array(compendium_1_support).astype(np.double)
compendium_2_support = np.array(compendium_2_support).astype(np.int)

plt.hist(np.log(compendium_2_support), 100)
plt.show()

plt.hist(np.log(compendium_1_support[:, 0]), 100)
plt.show()

plt.hist(compendium_1_support[:, 1], 100)
plt.show()