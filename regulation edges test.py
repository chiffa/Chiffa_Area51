from csv import reader as csv_reader
import numpy as np
from matplotlib import pyplot as plt
import os
import networkx as nx


prefix = 'C:\\Users\\Andrei\\Dropbox\\workspaces\\JHU\\Ewald Lab\\TF 2 targets'
loc_1 = os.path.join(prefix, 'human_cellnet_grn_Apr_05_2017.csv')
loc_2 = os.path.join(prefix, 'TRRUST\\trrust_rawdata.human.tsv')

marbach_prefix = 'C:\\Users\\Andrei\\Dropbox\\workspaces\\JHU\\Ewald Lab\\TF 2 targets\Marbach2016\\Network_compendium\\Tissue-specific_regulatory_networks_FANTOM5-v1\\32_high-level_networks'


def open_cellnet_grn(cellnet_file):
    compendium = []
    compendium_support = []
    with open(cellnet_file, 'rb') as source:
        reader = csv_reader(source, delimiter=',')
        header = reader.next()
        # print header
        for line in reader:
            interaction_no = int(line[0])
            interaction_from = line[1]
            interaction_to = line[2]
            interaction_z_score = float(line[3])
            interaction_correlation = float(line[4])
            compendium.append((interaction_from, interaction_to))
            compendium_support.append((interaction_z_score, interaction_correlation))

    return compendium, compendium_support


def open_TRRUST(trrust_file):
    compendium = []
    compendium_support = []
    with open(trrust_file, 'rb') as source:
        reader = csv_reader(source, delimiter='\t')
        for line in reader:
            interaction_from = line[0]
            interaction_to = line[1]
            interaction_type = line[2]
            evidence = line[3].split(';')
            evidence_redundancy = len(evidence)
            compendium.append((interaction_from, interaction_to))
            compendium_support.append((evidence_redundancy))

    return compendium, compendium_support


def compare_compendiums(compendium_1, compendium_1_support, compendium_2, compendium_2_support):

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


def open_marbach(marbach_file):
    compendium = []
    compendium_support = []
    with open(marbach_file, 'rb') as source:
        reader = csv_reader(source, delimiter='\t')
        for line in reader:
            interaction_from = line[0]
            interaction_to = line[1]
            if len(line) > 2:
                weight = float(line[3])
            else:
                weight = np.nan
            compendium.append((interaction_from, interaction_to))
            compendium_support.append((weight))


if __name__ == "__main__":
    # c1, c1_s = open_cellnet_grn(loc_1)
    # c2, c2_s = open_TRRUST(loc_2)
    # compare_compendiums(c1, c1_s, c2, c2_s)
    open_marbach()