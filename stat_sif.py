"""
Smoothing individual function without preselected bandwidth.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np
from numpy.linalg import inv
from stat_kernel import ep_kernel

"""
installed all the libraries above
"""


def sif(coord_mat, resy_design, h_opt):
    """
    Smoothing individual function without preselected bandwidth.

    Args:
        coord_mat (matrix): common coordinate matrix (l*d)
        resy_design (matrix): residual response matrix (n*l*m, m=d in MFSDA)
        h_opt (matrix): optimal bandwidth (m*d)
    """

    # Set up
    n, l, m = resy_design.shape
    d = coord_mat.shape[1]

    efit_eta = np.zeros((n, l, m))

    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((l, l, d + 1))  # L x L x d + 1 matrix
    t_mat0[:, :, 0] = np.ones((l, l))

    for dii in range(d):
        t_mat0[:, :, dii + 1] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))

    for mii in range(m):
        k_mat = np.ones((l, l))

        for dii in range(d):
            h = h_opt[mii, dii]
            k_mat = k_mat * ep_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function

        t_mat = np.transpose(t_mat0, [0, 2, 1])  # L x d+1 x L matrix

        for lii in range(l):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1))) * t_mat[:, :, lii]  # L0 x d+1 matrix
            sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii]) + np.eye(d + 1) * 0.0001)), kx.T)
            efit_eta[:, lii, mii] = np.squeeze(np.dot(resy_design[:, :, mii], sm_weight.T))

    res_eta = resy_design - efit_eta  # n x L x m matrix of difference between resy_design and fitted eta

    esig_eta = np.zeros((m, m, l))
    for lii in range(l):
        esig_eta[:, :, lii] = np.dot(np.squeeze(efit_eta[:, lii, :]).T, np.squeeze(efit_eta[:, lii, :]))/n

    return efit_eta, res_eta, esig_eta
