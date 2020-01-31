"""Module for tomography."""

import numpy as np
import tomorectv3d.tomorectv as tomorectv


def getp(a):
    return a.__array_interface__['data'][0]


class Solver(tomorectv):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    """

    def __init__(self, N, Ntheta, Nz, Nzp, method, ngpus, center, lambda0):
        """Please see help(Solver) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(N, Ntheta, Nz, Nzp, method, ngpus, center, lambda0)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def settheta(self, theta):
        super().settheta(getp(theta))

    def radon(self, data, f):
        super().radon_wrap(getp(np.array(data,order='C')), getp(np.array(f,order='C')))

    def itertvR(self, res, data, niter):
        super().itertvR_wrap(getp(np.array(res,order='C')), getp(np.array(data,order='C')), niter)
