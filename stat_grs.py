"""
Generating Wild Bootstrap samples.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np

"""
installed all the libraries above
"""


def grs(efity_design, efit_eta, res_eta):
    """
    Generating Wild Bootstrap samples.

    Args:
        efity_design (matrix): fitted response matrix (n*l*m, m=d in MFSDA)
        efit_eta (matrix): fitted individual curves (n*l*m, m=d in MFSDA)
        res_eta (matrix): residuals after correcting etas (n*l*m, m=d in MFSDA)
    """

    n, l, m = res_eta.shape
    simy_design = np.zeros((n, l, m))

    taus_tp = np.random.normal(0, 1, (n, l+1))
    taus = np.dot(np.atleast_2d(taus_tp[:, 0]).T, np.ones((1, l)))
    taumat = taus_tp[:, 1:(l+1)]

    for mii in range(m):
        simy_design[:, :, mii] = efity_design[:, :, mii] +\
                                 taus*np.squeeze(efit_eta[:, :, mii])+taumat*np.squeeze(res_eta[:, :, mii])

    return simy_design
