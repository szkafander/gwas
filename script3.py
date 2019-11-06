# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:12:01 2019

@author: tothp
"""

import ml
import numpy as np
import preprocessing


# prepare frequency data
data = preprocessing.preprocess_data("C:\\projects\\gwas")
frequencies = data["genotype"].count_alleles(
        data["allele_names"],
        standardize_alleles=True
    )
x = frequencies.loc[:, frequencies.columns != "UME_name"].to_numpy()

#%% plot variance along positions
var = np.std(data["genotype"], axis=0)

# train frequency network

# prepare snp data

# train snp network