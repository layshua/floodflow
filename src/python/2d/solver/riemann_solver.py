from abc import ABCMeta, abstractmethod


class RiemannSolverException(Exception):
    pass


class RiemannSolverSWE1D(object):
    """Abstract base class to provide interface to all
    subsequent one-dimensional Shallow Water Equation-
    based Riemann Solvers."""
    
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def solve(self, wavespeed=None):
        raise NotImplementedError(
            "Should implement solve(..)"
        )
