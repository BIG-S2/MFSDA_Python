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


def lpks_wb1(coord_mat, x_design, y_design, flag):
    """
    Local linear kernel smoothing for optimal bandwidth selection.

    Args:
        coord_mat (matrix): common coordinate matrix (l*d)
        x_design (matrix): design matrix (n*p)
        y_design (matrix): shape data (response matrix, n*l*m, m=d in MFSDA)
        flag (vector): indices for optimal bandwidth （m*1)
    """

    # Set up
    n, p = x_design.shape
    l, d = coord_mat.shape
    m = y_design.shape[2]
    efit_beta = np.zeros((p, l, m))
    efity_design = y_design * 0

    nh = 50  # the number of candidate bandwidth
    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((l, l, d + 1))  # L x L x d + 1 matrix
    t_mat0[:, :, 1] = np.ones((l, l))

    for dii in range(d):
        t_mat0[:, :, dii + 1] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))

    hat_mat = np.dot(inv(np.dot(x_design.T, x_design) + np.eye(p) * 0.0001), x_design.T)

    for mii in range(m):

        k_mat = np.ones((l, l))

        for dii in range(d):
            coord_range = np.ptp(coord_mat[:, dii])
            h_min = 0.01  # minimum bandwidth
            h_max = 0.5 * coord_range  # maximum bandwidth
            vh = np.logspace(np.log10(h_min), np.log10(h_max), nh)  # candidate bandwidth
            h = vh[int(flag[mii])]
            k_mat = k_mat * ep_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function

        t_mat = np.transpose(t_mat0, [0, 2, 1])  # L x d+1 x L matrix

        for lii in range(l):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[:, :, lii]  # L0 x d+1 matrix
            hat_y_design = np.dot(hat_mat, y_design[:, :, mii])
            sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii])+np.eye(d+1)*0.0001)), kx.T)
            efit_beta[:, lii, mii] = np.squeeze(np.dot(hat_y_design, sm_weight.T))

        efity_design[:, :, mii] = np.dot(x_design, efit_beta[:, :, mii])

    return efit_beta, efity_design
