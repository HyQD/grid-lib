import numpy as np
import abc

class PseudospectralGrid(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __repr__(self):
        pass
