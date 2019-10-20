# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:49:21 2019

@author: tothp
"""


import gwas
import os


folder = "C:\\projects\\gwas"
fname_genotypes = "popGen.txt"

snps = gwas.SNPs.from_file(os.path.join(folder, fname_genotypes))