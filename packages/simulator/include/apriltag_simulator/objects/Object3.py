from abc import abstractmethod
from apriltag_simulator.utils import transformations


class Object3:

    ITERATOR_STEP_M = 0.0001
    # ITERATOR_STEP_M = 0.0003

    def __init__(self, name, xyz=None, rpy=None):
        if xyz is None:
            xyz = [0, 0, 0]
        if rpy is None:
            rpy = [0, 0, 0]
        self._name = name
        self._xyz = xyz
        self._rpy = rpy

    @property
    def to_world_matrix(self):
        return transformations.compose_matrix(translate=self._xyz, angles=self._rpy)

    @abstractmethod
    def _get_point(self, x, y, z):
        raise NotImplementedError('The iterator method should be implemented by the child classes')

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError('The iterator method should be implemented by the child classes')
