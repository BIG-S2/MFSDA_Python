"""
calculating the test statistics.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np
from numpy.linalg import inv

"""
installed all the libraries above
"""


def wald_ht(x_design, efit_beta, esig_eta, cdesign):
    """
    Smoothing individual function without preselected bandwidth.

    Args:
        x_design (matrix): design matrix (n*p)
        efit_beta (matrix): coefficient matrix (p*l*m, m=d in MFSDA)
        esig_eta (matrix): covariance matrix of eta (m*m*l, m=d in MFSDA)
        cdesign (matrix): linear constraint matrix (1*(p-1))
    """

    # Set up
    p, l, m = efit_beta.shape
    delta_beta = np.zeros((m*p, l))

    for mii in range(m):
        delta_beta[(mii*p):(mii+1)*p, :] = efit_beta[:, :, mii]

    c_mat = np.kron(np.eye(m), cdesign)
    dd = np.dot(c_mat, delta_beta)
    omegax = inv(np.dot(cdesign, np.dot(inv(np.dot(x_design.T, x_design)), cdesign.T)))
    lstat = np.zeros((l, 1))

    for lii in range(l):
        inv_esig_eta = inv(np.squeeze(esig_eta[:, :, lii]))
        lstat[lii] = np.dot(np.dot(dd[:, lii].T, inv_esig_eta), dd[:, lii])

    lstat = omegax * lstat
    gstat = np.mean(lstat)

    return gstat, lstat
