# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:49:21 2019

@author: tothp
"""


import gwas
import numpy as np
import os
import pandas as pd

from typing import List


path_project = "C:\\projects\\gwas"
fn_genotypes = "popGen.txt"
fn_phenotypes = "AM_BV_Combined.csv"
fn_matching = "AM_Seq_SkodIDs.txt"


def preprocess_data(
        path_project: str,
        fn_genotypes: str = "popGen.txt",
        fn_phenotypes: str = "AM_BV_Combined.csv",
        fn_matching: str = "AM_Seq_SkodIDs.txt"
    ) -> List:
    snps = gwas.SNPs.from_file(os.path.join(path_project, fn_genotypes))
    
    # we need a matching x table and 9 matching y tables
    delimiter = "\t"
    
    # read matching table
    print("Reading genotype data...")
    phenotype_genotype_matching = pd.read_csv(
            os.path.join(path_project, fn_matching),
            delimiter=delimiter,
            header=None
        )
    print("Genotype data read.")
    
    # read phenotype data
    print("Reading phenotype data...")
    phenotype = pd.read_csv(
            os.path.join(path_project, fn_phenotypes),
            delimiter=";"
        )
    print("Phenotype data read.")

    # put into wide format wrt. variable
    # need 9 X n_samples X n_rings tables
    print("Matching genotype and phenotype entries...")
    traits = {
            trait: phenotype[phenotype["Trait"] == trait].filter(
                    regex="Ring|Mum"
                ) for trait in phenotype["Trait"].unique()
            }
    
    # there are 521 traits but 526 SNPs. must filter SNPs.
    # find sample names in SNPs that match the 521 traits.
    # check if all Mum_names are the same in all 9 trait tables
    unique_names = set()
    for key in traits.keys():
        names = traits[key]["Mum_name"]
        unique_names.update(set(names))
    assert len(unique_names) == len(traits[key]["Mum_name"]), \
        "Sample S names not consistent" # OK
    
    # now the relevant sample names are unique_names
    # get overlap between matching and traits sample names ("S names")
    inds_overlap_y = phenotype_genotype_matching[1].isin(unique_names)
    
    # check overlap
    assert phenotype_genotype_matching[inds_overlap_y][1].isin(
            unique_names).all(), "Sample S names incomplete overlap"

    # get overlap between matching and snps sample names ("UME names")
    inds_overlap_x = phenotype_genotype_matching[0].isin(snps.sample_names)
    assert phenotype_genotype_matching[inds_overlap_x][0].isin(
            snps.sample_names).all(), "Sample UME names incomplete overlap"

    # samples that are both in x and y and can be matched
    inds_overlap = np.logical_and(inds_overlap_x.to_numpy(),
                                  inds_overlap_y.to_numpy())
    phenotype_genotype_matching = phenotype_genotype_matching[inds_overlap]
    
    # check overlap
    assert phenotype_genotype_matching[1].isin(unique_names).all(), \
        "Sample S names incomplete overlap"
    assert phenotype_genotype_matching[0].isin(snps.sample_names).all(), \
        "Sample UME names incomplete overlap"

    # we must reorder x and all y's so that they are in the same order
    
    # find inds of UME names in snps in the same order as they appear in
    # phenotype_genotype_matching[0]
    ume_names_matchable = phenotype_genotype_matching[0].to_list()
    ume_names_all = list(snps.sample_names)
    inds_ume = [ume_names_all.index(name) for name in ume_names_matchable]
    
    snps_matched = snps[inds_ume]
    
    # find inds of S names in all trait tables in the same order as they appear
    # in phenotype_genotype_matching[1]
    traits_matched = dict()
    s_names_matchable = phenotype_genotype_matching[1].to_list()
    for trait_name, trait_vals in traits.items():
        s_names_all = trait_vals["Mum_name"].to_list()
        inds_s = [s_names_all.index(name) for name in s_names_matchable]
        traits_matched[trait_name] = traits[trait_name].iloc[inds_s]
    print("Genotype and phenotype entries matched.")

    # many SNP samples are unusable - get rid of these
    # start by removing samples that have too many missing measurements
    # remove these from traits as well
    
    # step 1 - remove samples with missing measurements
    print("Filtering data...")
    snps_filtered, inds = snps_matched.valid_samples(
            max_invalid_alleles=100000,
            return_indices=True
        )

    # do the same for traits
    traits_filtered = dict()
    for variable in traits_matched:
        traits_filtered[variable] = traits_matched[variable].iloc[inds]
    
    # step 2 - get relevant allele names
    allele_names_relevant = [a for a in snps_filtered.unique_alleles
                             if not gwas.is_wildcard(a)]
    allele_inds_relevant = [snps_filtered.name_to_ind[a] for a
                            in allele_names_relevant]
    print("Data filtered.")

    return {
            "genotype": snps_filtered,
            "phenotype": traits_filtered,
            "allele_names": allele_names_relevant,
            "allele_inds": allele_inds_relevant
           }