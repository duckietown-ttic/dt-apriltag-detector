import numpy as np

from abc import abstractmethod
from apriltag_simulator.utils import transformations


class Object3:

    DEFAULT_STEP_M = 0.0001

    def __init__(self, name, xyz=None, rpy=None):
        if xyz is None:
            xyz = [0, 0, 0]
        if rpy is None:
            rpy = [0, 0, 0]
        self._name = name
        self._xyz = list(xyz)
        self._rpy = list(rpy)
        # create obj_frame -> world transformation
        self._to_world = transformations.compose_matrix(translate=self._xyz, angles=self._rpy)
        self._from_world = transformations.inverse_matrix(self._to_world)

    @property
    def name(self):
        return self._name

    @property
    def to_world_matrix(self):
        return self._to_world

    @property
    def from_world_matrix(self):
        return self._from_world

    def transform_to_world(self, p):
        if isinstance(p, np.ndarray):
            p = p.tolist()
        if len(p) == 3:
            p = p + [1]
        return np.dot(self.to_world_matrix, p)[0:3]

    def transform_from_world(self, p):
        if isinstance(p, np.ndarray):
            p = p.tolist()
        if len(p) == 3:
            p = p + [1]
        return np.dot(self.from_world_matrix, p)[0:3]

    @abstractmethod
    def shadow_polygon(self):
        raise NotImplementedError('The `shadow_polygon` method must be implemented by child classes')

    @abstractmethod
    def points(self, steps_x=None, steps_y=None):
        raise NotImplementedError('The `points` method must be implemented by child classes')

    @abstractmethod
    def _get_point(self, x, y, z):
        raise NotImplementedError('The `_get_point` method must be implemented by child classes')
