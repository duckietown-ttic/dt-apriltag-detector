from .Object3 import Object3

import numpy as np


class Rectangle3(Object3):

    def __init__(self, name, dimensions=None, xyz=None, rpy=None):
        Object3.__init__(self, name, xyz, rpy)
        if dimensions is None:
            dimensions = [1, 1]
        self._dimensions = dimensions

    def _get_point(self, x, y, *_):
        return np.array([(x - 0.5) * self._dimensions[0], (y - 0.5) * self._dimensions[1], 0])

    def _optimal_step(self):
        return self.ITERATOR_STEP_M * max(1, 20 * self._xyz[2])

    def __enumerated_iter__(self):
        steps = np.array(self._dimensions) / self._optimal_step()
        world = self.to_world_matrix
        for i in np.linspace(0, 1, steps[0]):
            for j in np.linspace(0, 1, steps[1]):
                p = np.dot(world, np.array(list(self._get_point(i, j, 0).tolist()) + [1]))
                yield (i, j), p[0:3]

    def __iter__(self):
        for _, p in self.__enumerated_iter__():
            yield p
