"""
Local linear kernel smoothing for optimal bandwidth selection.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np
from numpy.linalg import inv
from stat_kernel import ep_kernel

"""
installed all the libraries above
"""


def lpks_wob(coord_mat, x_design, y_design):
    """
    Local linear kernel smoothing for optimal bandwidth selection.

    Args:
        coord_mat (matrix): common coordinate matrix (l*d)
        x_design (matrix): design matrix (n*p)
        y_design (matrix): shape data (response matrix, n*l*m, m=d in MFSDA)
    """

    # Set up
    n, p = x_design.shape
    l, d = coord_mat.shape
    m = y_design.shape[2]
    efity_design = y_design * 0

    nh = 50      # the number of candidate bandwidth
    k = 5  # number of folders
    k_ind = np.floor(np.linspace(1, n, k + 1))
    gcv = np.zeros((nh, m))      # GCV performance function

    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((l, l, d + 1))  # L x L x d + 1 matrix
    t_mat0[:, :, 1] = np.ones((l, l))

    for dii in range(d):
        t_mat0[:, :, dii + 1] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))

    for nhii in range(nh):
        k_mat = np.ones((l, l))

        for dii in range(d):
            coord_range = np.ptp(coord_mat[:, dii])
            h_min = 0.01  # minimum bandwidth
            h_max = 0.5 * coord_range  # maximum bandwidth
            vh = np.logspace(np.log10(h_min), np.log10(h_max), nh)  # candidate bandwidth
            h = vh[nhii]
            k_mat = k_mat * ep_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function

        t_mat = np.transpose(t_mat0, [0, 2, 1])  # L x d+1 x L matrix

        for mii in range(m):
            for lii in range(l):
                kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[:, :, lii]  # L0 x d+1 matrix
                indx = np.ones((1, n))
                for kii in range(k):
                    ind_beg = int(k_ind[kii]-1)
                    ind_end = int(k_ind[kii+1])
                    indx[0, ind_beg:ind_end] = 0
                    x_design0 = x_design[np.nonzero(indx == 1)[0], :]
                    hat_mat = np.dot(inv(np.dot(x_design0.T, x_design0)+np.eye(p)*0.0001), x_design0.T)
                    hat_y_design0 = np.dot(hat_mat, y_design[np.nonzero(indx == 1)[0], :, mii])
                    sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii])+np.eye(d+1)*0.0001)), kx.T)
                    efit_beta = np.dot(hat_y_design0, sm_weight.T)
                    efity_design[ind_beg:ind_end, lii, mii] = np.squeeze(np.dot(x_design[ind_beg:ind_end, :], efit_beta))

            gcv[nhii, mii] = np.mean((y_design[:, :, mii]-efity_design[:, :, mii])**2)

    flag = np.argmin(gcv, axis=0)  # optimal bandwidth

    return flag
