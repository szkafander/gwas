# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:57:30 2019

@author: tothp
"""


import matplotlib.pyplot as pl
import numpy as np
import pandas as pd


def plot_share_of_frequencies(
        counts: pd.DataFrame,
        name: str = "UME_name"
    ) -> None:
    k = 0
    x = np.arange(len(counts))
    vals_old = None
    for name, vals in counts.iteritems():
        if name != "UME_name":
            if k == 0:
                bottom = np.zeros(vals.shape)
            else:
                bottom = bottom + vals_old
            pl.bar(x, vals, 1, bottom=bottom)
            vals_old = vals
            k = k + 1
    pl.xticks(x, labels=counts[name], rotation=-90, fontsize=8)
    pl.ylim([0, 1])
    pl.xlim([x.min(), x.max()])
