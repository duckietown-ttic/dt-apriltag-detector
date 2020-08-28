from .CameraLens import CameraLens

import numpy as np


class FishEyeLens(CameraLens):

    def __init__(self, name, k1=0, k2=0, k3=0, p1=0, p2=0):
        CameraLens.__init__(self, name)
        self._name = name
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2

    def transform(self, point2):
        x, y = point2.tolist()
        r = np.linalg.norm(point2)
        radial = (1 + self._k1 * (r ** 2) + self._k2 * (r ** 4) + self._k3 * (r ** 6)) * point2
        tangential = np.array([
            2 * self._p1 * x * y + self._p2 * ((r ** 2) + 2 * (x ** 2)),
            2 * self._p2 * x * y + self._p1 * ((r ** 2) + 2 * (y ** 2))
        ])
        # TODO: remove this
        # return radial
        return np.add(radial, tangential)
