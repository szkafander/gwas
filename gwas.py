# -*- coding: utf-8 -*-
"""
Classes for base GWAS infrastructure.
"""


import numpy as np
import pandas as pd
import warnings

from typing import Callable, List, Optional, Tuple


def is_wildcard(allele: str) -> bool:
    return "*" in allele or "." in allele or "," in allele


def is_valid(allele: str) -> bool:
    return not is_wildcard(allele)


allele_missing = "./."


class SNPs:
    """ This subclasses numpy.ndarray. You can slice, logical index, etc., just
    like with numpy arrays. All slices return SNPs objects. This class has
    support for reading, filtering and comparing SNP data.

    Why not DataFrame immediately? I found that counting allele occurrences in 
    numpy arrays is much faster than in DataFrames. You can always cast SNPs 
    into DataFrames using the .to_dataframe method. The method .count_alleles 
    return DataFrames.

    """

    def __init__(
            self,
            data: np.ndarray,
            sample_names: Optional[np.ndarray] = None,
            positions: Optional[np.ndarray] = None,
            unique_alleles: Optional[np.ndarray] = None,
            name_to_ind: Optional[dict] = None,
            ind_to_name: Optional[dict] = None
        ) -> None:
        self.data = data
        self.sample_names = sample_names
        self.positions = positions
        self.unique_alleles = unique_alleles
        self.name_to_ind = name_to_ind
        self.ind_to_name = ind_to_name

    def __getitem__(self, index) -> "SNPs":
        return SNPs(
                self.data[index],
                self.sample_names[index],
                self.positions,
                self.unique_alleles,
                self.name_to_ind,
                self.ind_to_name
            )

    def __len__(self) -> int:
        n = len(self.sample_names)
        assert n == len(self.data), \
            "Number of sample names must equal length of data."
        return n

    @property
    def metadata(self) -> dict:
        return {
                "sample_names": self.sample_names,
                "positions": self.positions,
                "unique_alleles": self.unique_alleles,
                "name_to_ind": self.name_to_ind,
                "ind_to_name": self.name_to_ind
            }

    def copy(self, data: Optional[np.ndarray] = None) -> "SNPs":
        if data is None:
            return SNPs(self.data, **self.metadata)
        return SNPs(data, **self.metadata)

    @staticmethod
    def _split_line(line: str, delimiter: str = "\t") -> Tuple[str, List]:
        items = line.split(delimiter)
        return "_".join(items[:3]), items[3:]

    @staticmethod
    def _get_info(path: str, delimiter: str = "\t") -> int:
        """ Returns number of samples, number of SNP's and unique allele names.
        Assumes data format that was sent. 

        """
        header = open(path).readline().split(delimiter)
        sample_names = header[3:]
        unique_alleles = set()
        n_samples = len(sample_names)
        n_snps = -1 # -1 for the header
        with open(path) as file:
            for k, line in enumerate(file):
                if k != 0:
                    line = line.strip()
                    _, alleles = SNPs._split_line(line)
                    unique_alleles.update(alleles)
                n_snps += 1
        return n_samples, n_snps, unique_alleles

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
                        warnings.warn("Data inconsistent:", RuntimeWarning)
                        warnings.warn("First inconsistent row: {}.".format(k),
                                      RuntimeWarning)
        return consistent

    @staticmethod
    def from_file(path: str, delimiter: str = "\t") -> "SNPs":
        """ Instantiates an SNPs object from a data file. The data file is a
        delimited text file holding tabular SNP data. The rows are the alleles,
        the columns are samples. 

        """
        n_samples, n_snps, unique_alleles = SNPs._get_info(path, delimiter)
        name_to_ind = {allele: i for i, allele in enumerate(unique_alleles)}
        snps = np.zeros((n_samples, n_snps))
        positions = []
        if SNPs._check_row_col_consistency(path, delimiter):
            # load values here
            # this can be a large array - only use if n_samples X n_snps is
            # manageable. loading data can take a while.
            with open(path) as file:
                for k, line in enumerate(file):
                    line = line.strip()
                    # skip header
                    if k != 0:
                        position, alleles = SNPs._split_line(line)
                        positions.append(position)
                        snps[:, k - 1] = [name_to_ind[i] for i in alleles]
                    else:
                        _, sample_names = SNPs._split_line(line)
                    # log progress so we know that something is happening
                    if k % 10000 == 0:
                        print("\r{}/{} lines processed.".format(k, n_snps),
                              flush=True, end="")
            return SNPs(
                    snps,
                    np.array(sample_names),
                    np.array(positions),
                    unique_alleles,
                    name_to_ind,
                    {val: key for key, val in name_to_ind.items()}
                )
        else:
            raise IOError("Data file inconsistent.")

    def to_names(self) -> "SNPs":
        lut = np.fromiter(self.ind_to_name.values(), dtype=np.str)
        return self.copy(lut[self])

    def to_inds(self) -> "SNPs":
        pass

    def filter_samples(
            self,
            criterion: Callable,
            return_indices: bool = False
        ) -> "SNPs":
        inds = np.apply_along_axis(criterion, 1, self.data)
        if return_indices:
            return self[inds], inds
        return self[inds]

    def valid_samples(
            self,
            max_invalid_alleles: int = 1000000,
            return_indices: bool = False
        ) -> "SNPs":
        """ Filters SNP's based on maximum number of admissible invalid
        alleles.

        """
        ind_missing = self.name_to_ind[allele_missing]
        def criterion(sample: np.ndarray) -> int:
            return np.sum(sample == ind_missing) <= max_invalid_alleles
        return self.filter_samples(criterion, return_indices=return_indices)

    def normalize_alleles(self, mode: str = "length") -> "SNPs":
        if mode == "length":
            return self.copy(self.data / len(self.data))
        elif mode == "sum":
            return self.copy(self.data / self.data.sum(axis=0))
        else:
            raise ValueError("Mode must be either 'length' or 'sum'.")

    def entropy_of_alleles(self) -> np.ndarray:
        px = self.normalize_alleles().data
        return - np.sum(px * np.log2(px), axis=0)

    def count_alleles(
            self,
            alleles: List[str],
            normalize_alleles: bool = False,
            normalize_samples: bool = False,
            standardize_alleles: bool = False
        ) -> pd.DataFrame:
        """ Counts the occurrence of alleles for each sample. Can normalize 
        sample-wise.

        """
        n = len(self)
        counts = np.zeros((n, len(alleles)))
        for k, allele in enumerate(alleles):
            occurrences = np.sum(self.data == self.name_to_ind[allele], axis=1)
            if normalize_alleles:
                occurrences = occurrences / occurrences.sum()
            counts[:, k] = occurrences
        if normalize_samples:
            counts = counts / counts.sum(axis=1, keepdims=True)
        if standardize_alleles:
            counts = counts - counts.mean(axis=0, keepdims=True)
            counts = counts / counts.std(axis=0, keepdims=True)
        df = pd.DataFrame(data=counts,
                          index=np.arange(len(self.sample_names)),
                          columns=alleles)
        df["UME_name"] = self.sample_names
        return df

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.data,
                          index=np.arange(len(self.sample_names)),
                          columns=self.positions)
        df["UME_Name"] = self.sample_names
        return df
