# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:28:25 2019

@author: tothp
"""
#%%
import numpy as np
import matplotlib.pyplot as pl
import os


folder = "C:\\projects\\gwas"

fname_genotypes = "popGen.txt"
fname_phenotypes = "AM_BV_Combined.csv"
fname_matching = "AM_Seq_SkodIDs.txt"


import pandas as pd

phenotype = pd.read_csv(os.path.join(folder, fname_phenotypes), delimiter=";")
# phenotype features of samples, mean over rings
# this is in the order of mum name
traits = {
        trait: phenotype[phenotype["Trait"] == trait].filter(regex="Ring")
        for trait in phenotype["Trait"].unique()
        }

# this is in the order of mum name
delimiter = "\t"

phenotype_genotype_matching = pd.read_csv(
        os.path.join(folder, fname_matching), 
        delimiter=delimiter,
        header=None
    ).sort_values(1)


def get_alleles(line: str):
    return line.split(delimiter)[3:]


def get_n_snps(path: str):
    k = -1 # -1 for the header
    with open(path) as file:
        for line in file:
            k += 1
    return k


def check_row_col_consistency(path: str):
    with open(path) as file:
        for k, line in enumerate(file):
            if k == 0:
                n_cols = len(line.split(delimiter))
            else:
                if len(line.split(delimiter)) != n_cols:
                    print("data inconsistent")
                    print("first inconsistent row: {}".format(k))
        print("data consistent")
    return None


def get_unique_alleles(path: str):
    unique_alleles = set()
    with open(path) as file:
        for k, line in enumerate(file):
            if k != 0:
                line = line.strip()
                alleles = get_alleles(line)
                unique_alleles.update(alleles)
    return unique_alleles


def is_wildcard(allele: str):
    return "*" in allele or "." in allele or "," in allele


def is_valid(allele: str):
    return not is_wildcard(allele)

#%%
path_genotypes = os.path.join(folder, fname_genotypes)
n_snps = get_n_snps(path_genotypes)
check_row_col_consistency(path_genotypes)

#%%
header = open(path_genotypes).readline().split("\t")
sample_names = header[3:]
n_samples = len(sample_names)
unique_alleles = get_unique_alleles(path_genotypes)

#%%
wildcard_inds = np.nonzero([is_wildcard(a) for a in unique_alleles])[0]

#%%

ind_allele_mapping = {allele: i for i, allele in enumerate(unique_alleles)}
#%%
genotypes = np.zeros((n_samples, n_snps))

with open(path_genotypes) as file:
    for k, line in enumerate(file):
        if k != 0:
            line = line.strip()
            alleles = get_alleles(line)
            genotypes[:, k - 1] = [ind_allele_mapping[i] for i in alleles]
        if k % 10000 == 0:
            print("{} lines processed".format(k))

#%% ./. is some missing value, get the number of these per row

missing_ind = 28
n_missing = np.sum(genotypes == ind_allele_mapping["./."], axis=1)
pl.plot(n_missing)
pl.xlabel("sample no.")
pl.ylabel("number of missing measurements")

#%% get the number of samples with enough valid measurements

ind_valid = [val for key, val in ind_allele_mapping.items() if is_valid(key)]
