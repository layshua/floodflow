from abc import ABCMeta, abstractmethod


class RiemannSolverException(Exception):
    pass


class RiemannSolver2D(object):
    """Abstract base class to provide interface to all
    subsequent two-dimensional Shallow Water Equation-
    based Riemann Solvers."""
    
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def flux(self):
        raise NotImplementedError(
            "Should implement flux()"
        )
