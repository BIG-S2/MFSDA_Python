"""
Local linear kernel smoothing with optimal bandwidth selection (Silverman's rule of thumb).

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-24
"""

import numpy as np
from numpy.linalg import inv
from stat_kernel import ep_kernel

"""
installed all the libraries above
"""


def lpks(coord_mat, x_design, y_design):
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
    efit_beta = np.zeros((p, l, m))
    efity_design = np.zeros((n, l, m))

    nh = 50  # the number of candidate bandwidth
    gcv = np.zeros((nh, m))  # GCV performance function

    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((l, l, d + 1))  # L x L x d + 1 matrix
    t_mat0[:, :, 0] = np.ones((l, l))
    vh = np.zeros((nh, d))
    h_opt = np.zeros((m, d))

    for dii in range(d):
        t_mat0[:, :, dii + 1] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))
        coord_range = np.ptp(coord_mat[:, dii])
        h_min = 0.01  # minimum bandwidth
        h_max = 0.5 * coord_range  # maximum bandwidth
        vh[:, dii] = np.logspace(np.log10(h_min), np.log10(h_max), nh)  # candidate bandwidth

    t_mat = np.transpose(t_mat0, [0, 2, 1])  # L x d+1 x L matrix
    hat_mat = np.dot(inv(np.dot(x_design.T, x_design) + np.eye(p) * 0.00001), x_design.T)

    for nhii in range(nh):
        k_mat = np.ones((l, l))
        for dii in range(d):
            h = vh[nhii, dii]
            k_mat = k_mat * ep_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function
        for mii in range(m):
            hat_y_design = np.dot(hat_mat, y_design[:, :, mii])
            for lii in range(l):
                kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[:, :, lii]  # L0 x d+1 matrix
                sm_weight0 = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii]) + np.eye(d + 1) * 0.00001)), kx.T)
                efity_design[:, lii, mii] = np.squeeze(np.dot(x_design, np.dot(hat_y_design, sm_weight0.T)))
            gcv[nhii, mii] = np.mean((y_design[:, :, mii]-efity_design[:, :, mii])**2)

    flag = np.argmin(gcv, axis=0)  # optimal bandwidth

    for mii in range(m):

        k_mat = np.ones((l, l))
        hat_y_design = np.dot(hat_mat, y_design[:, :, mii])

        for dii in range(d):
            h = vh[int(flag[mii]), dii]
            k_mat = k_mat * ep_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function
            h_opt[dii, mii] = h

        for lii in range(l):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[:, :, lii]  # L0 x d+1 matrix
            sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii])+np.eye(d+1)*0.00001)), kx.T)
            efit_beta[:, lii, mii] = np.squeeze(np.dot(hat_y_design, sm_weight.T))

        efity_design[:, :, mii] = np.dot(x_design, efit_beta[:, :, mii])

    return efit_beta, efity_design, h_opt
