from .Object3 import Object3
from .utils import isect_line_plane_v3

import numpy as np


class Rectangle3(Object3):

    def __init__(self, name, dimensions=None, color=255, xyz=None, rpy=None):
        Object3.__init__(self, name, xyz, rpy)
        if dimensions is None:
            dimensions = [1, 1]
        self._dimensions = dimensions
        self._color = color if isinstance(color, (list, tuple)) else (
            [color, color, color] if isinstance(color, int) else [255] * 3)
        self._plane_center = self.transform_to_world(self._get_point(0.5, 0.5, 0))
        _plane_normal = self.transform_to_world([0, 0, -1]) - self._plane_center
        self._plane_normal = _plane_normal / np.linalg.norm(_plane_normal)

    def shadow_polygon(self):
        poly = np.array([
            self.transform_to_world(self._get_point(*c))
            for c in [(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)]
        ])
        # project polygon on the plane: z = min p.z   for every p in polygon
        poly[:, 2] = np.min(poly[:, 2])
        # ---
        return poly

    def num_points(self, steps_x=None, steps_y=None):
        if steps_x is None:
            steps_x = self._dimensions[0] / self._optimal_step()
        if steps_y is None:
            steps_y = self._dimensions[1] / self._optimal_step()
        # ---
        return steps_x * steps_y

    def points(self, steps_x=None, steps_y=None):
        if steps_x is None:
            steps_x = self._dimensions[0] / self._optimal_step()
        if steps_y is None:
            steps_y = self._dimensions[1] / self._optimal_step()
        # ---
        for i in np.linspace(0, 1, int(steps_x)):
            for j in np.linspace(0, 1, int(steps_y)):
                point3w = self.transform_to_world(self._get_point(i, j, 0))
                yield point3w, self._color

    def intersect(self, ray3w):
        camera_center = [0, 0, 0]
        intersection_w = \
            isect_line_plane_v3(camera_center, ray3w, self._plane_center, self._plane_normal)
        intersection = self.transform_from_world(intersection_w)
        if abs(intersection[0]) <= self._dimensions[0] * 0.5 and \
                abs(intersection[1]) <= self._dimensions[1] * 0.5:
            return intersection_w, self._color
        return None, None

    def _get_point(self, x, y, *_):
        return np.array([(x - 0.5) * self._dimensions[0], (y - 0.5) * self._dimensions[1], 0])

    def _optimal_step(self):
        return self.DEFAULT_STEP_M * max(1, 20 * self._xyz[2])
