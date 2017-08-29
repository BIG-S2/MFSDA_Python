"""
calculating p value using bootstrap sampling.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np
from scipy import stats
from stat_lpks import lpks
from stat_lpks_pre_bw import lpks_pre_bw
from stat_sif import sif
from stat_grs import grs
from stat_wald_ht import wald_ht
# from sklearn.cluster import KMeans
# from stat_gap import gap

"""
installed all the libraries above
"""


def bstrp_pvalue(coord_mat, x_design, y_design, cdesign, gstat, num_bstrp, thres, area):
    """
    Smoothing individual function without preselected bandwidth.

    Args:
        coord_mat (matrix): common coordinate matrix (l*d)
        x_design (matrix): design matrix (n*p)
        y_design (matrix): shape data (response matrix, n*l*m, m=d in MFSDA)
        cdesign (matrix): linear constraint matrix (1*(p-1))
        gstat (scalar): global test statistic
        num_bstrp (scalar): number of bootstrap
        thres (scalar): thresholding for clustering
        area (scalar): area of the largest connected region
    """

    # under the null hypothesis
    x_design0 = x_design[:, np.nonzero(cdesign == 0)[0]]

    efit_beta0, efity_design0, h_opt0 = lpks(coord_mat, x_design0, y_design)
    resy_design0 = y_design - efity_design0
    efit_eta0, res_eta0, esig_eta0 = sif(coord_mat, resy_design0, h_opt0)

    # Bootstrap procedures
    gstatvec = np.zeros((1, num_bstrp))
    simlpval_area = np.zeros((1, num_bstrp))

    for gii in range(num_bstrp):
        simy_design = grs(efity_design0, efit_eta0, res_eta0)
        simefit_beta = lpks_pre_bw(coord_mat, x_design, simy_design, h_opt0)[0]
        sim_gstat, sim_lstat = wald_ht(x_design, simefit_beta, esig_eta0, cdesign)
        gstatvec[0, gii] = sim_gstat

        sim_lpval = 1 - stats.chi2.cdf(sim_lstat, simy_design.shape[2])
        sim_ind_thres = sim_lpval <= 10**(-thres)
        simlpval_area[0, gii] = np.sum(sim_ind_thres)

    k1 = np.mean(gstatvec)
    k2 = np.var(gstatvec)
    k3 = np.mean((gstatvec - k1)**3)
    a = k3/(4*k2)
    b = k1-2*k2**2/k3
    d = 8*k2**3/(k3**2)
    gpval = 1 - stats.chi2.cdf((gstat - b)/a, d)

    clu_pval = np.sum(simlpval_area >= area)/num_bstrp

    return gpval, clu_pval
