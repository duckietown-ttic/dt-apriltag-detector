import numpy as np

from apriltag_simulator.lenses import CameraLens


class Camera(object):

    def __init__(self, name, fx, fy, cx, cy, width, height):
        self._name = name
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._C = np.array([
            [self._fx,         0,    self._cx],
            [0,         self._fy,    self._cy],
            [0,                0,           1]
        ])
        self._IO = np.array([
            [1,    0,    0,    0],
            [0,    1,    0,    0],
            [0,    0,    1,    0]
        ])
        self._width = width
        self._height = height
        self._lens = None

    def attach_lens(self, lens):
        if not isinstance(lens, CameraLens):
            raise ValueError(
                'Object `lens` must be of type CameraLens, got %s instead' % str(type(lens)))
        self._lens = lens

    def detach_lens(self):
        self._lens = None

    def capture(self, point3):
        if isinstance(point3, np.ndarray):
            point3 = point3.tolist()
        point3 = np.array(point3 + [1])
        point2 = np.matmul(np.matmul(self._C, self._IO), point3)
        point2 = point2[0:2] / point2[2]

        # apply lens distortion
        if self._lens is not None:

            xn = point3[0:2] / point3[2]

            dpoint2 = self._lens.transform(xn)

            point2 = np.matmul(self._C, dpoint2.tolist() + [1])
            point2 = point2[0:2]

        # ---
        return point2
