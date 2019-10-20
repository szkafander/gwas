# -*- coding: utf-8 -*-
"""
Classes for base GWAS infrastructure.
"""

import numpy as np
import warnings

from typing import Optional


class SNPs(np.ndarray):
    """ This subclasses numpy.ndarray. You can slice, logical index, etc., just
    like with numpy arrays. All slices return SNPs objects. This class has
    support for reading, filtering and comparing SNP data. """

    def __new__(
            clss,
            input_array: np.ndarray,
            allele_names: Optional[dict] = None,
            info: Optional[str] = None
        ) -> None:
        obj = np.asarray(input_array).view(clss)
        obj.allele_names = allele_names
        obj.info = info
        return obj

    def __array_finalize__(self, obj: np.ndarray) -> None:
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    @staticmethod
    def _get_info(path: str, delimiter: str = "\t") -> int:
        """ Returns number of samples and number of SNP's. Assumes data format 
        that was sent. """
        header = open(path).readline().split(delimiter)
        sample_names = header[3:]
        n_samples = len(sample_names)
        n_snps = -1 # -1 for the header
        with open(path) as file:
            for line in file:
                n_snps += 1
        return n_samples, n_snps

    @staticmethod
    def _get_alleles(line: str, delimiter: str = "\t"):
        return line.split(delimiter)[3:]

    @staticmethod
    def _get_unique_alleles(path: str):
        unique_alleles = set()
        with open(path) as file:
            for k, line in enumerate(file):
                if k != 0:
                    line = line.strip()
                    alleles = SNPs._get_alleles(line)
                    unique_alleles.update(alleles)
        return unique_alleles

    @staticmethod
    def _check_row_col_consistency(path: str, delimiter: str = "\t") -> bool:
        consistent = True
        with open(path) as file:
            for k, line in enumerate(file):
                if k == 0:
                    n_cols = len(line.split(delimiter))
                else:
                    if len(line.split(delimiter)) != n_cols:
                        consistent = False
                        warnings.warn("Data inconsistent:")
                        warnings.warn("First inconsistent row: {}.".format(k))
        return consistent

    @staticmethod
    def from_file(path: str, delimiter: str = "\t") -> "SNPs":
        unique_alleles = SNPs._get_unique_alleles(path)
        allele_names = {allele: i for i, allele in enumerate(unique_alleles)}
        n_samples, n_snps = SNPs._get_info(path, delimiter)
        snps = np.zeros((n_samples, n_snps))
        if SNPs._check_row_col_consistency(path, delimiter):
            # load values here
            # this can be a large array - only use if n_samples X n_snps is
            # manageable. loading data can take a while.
            with open(path) as file:
                for k, line in enumerate(file):
                    # skip header
                    if k != 0:
                        line = line.strip()
                        alleles = SNPs._get_alleles(line)
                        snps[:, k - 1] = [allele_names[i] for i in alleles]
                    # log progress so we know that something is happening
                    if k % 10000 == 0:
                        print("{}/{} lines processed".format(k, n_snps))
            snps = SNPs(snps)
            snps.unique_alleles = unique_alleles
            snps.allele_names = allele_names
            return snps
        else:
            raise IOError("Data file inconsistent.")
