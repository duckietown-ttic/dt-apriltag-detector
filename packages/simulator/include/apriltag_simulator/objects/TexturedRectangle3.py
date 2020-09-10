import os
import numpy as np
from imageio import imread

from .Rectangle3 import Rectangle3
from .utils import isect_line_plane_v3


class TexturedRectangle3(Rectangle3):

    def __init__(self, name, texture, dimensions, xyz=None, rpy=None):
        Rectangle3.__init__(self, name, dimensions, 0, xyz, rpy)
        self._texture_file = texture
        if not os.path.exists(self._texture_file) or not os.path.isfile(self._texture_file):
            raise ValueError('Could not load texture file "%s"' % self._texture_file)
        self._texture = imread(self._texture_file)[:, :, 0:3].transpose((1, 0, 2))

    def points(self, steps_x=None, steps_y=None):
        if steps_x is None:
            steps_x = self._dimensions[0] / self._optimal_step()
        if steps_y is None:
            steps_y = self._dimensions[1] / self._optimal_step()
        # ---
        for i in np.linspace(0, 1, int(steps_x)):
            for j in np.linspace(0, 1, int(steps_y)):
                point3w = self.transform_to_world(self._get_point(i, j, 0))
                color3 = self._get_uv_mapping(i, j)
                yield point3w, color3

    def intersect(self, ray3w):
        camera_center = [0, 0, 0]
        intersection_w = isect_line_plane_v3(camera_center, ray3w, self._plane_center, self._plane_normal)
        intersection = self.transform_from_world(intersection_w)
        if abs(intersection[0]) <= self._dimensions[0] * 0.5 and \
                abs(intersection[1]) <= self._dimensions[1] * 0.5:
            u = 0.5 + (intersection[0] / self._dimensions[0])
            v = 0.5 + (intersection[1] / self._dimensions[1])
            color3 = self._get_uv_mapping(u, v)
            return intersection_w, color3
        return None, None

    def _get_uv_mapping(self, x, y, *_):
        w, h = self._texture.shape[0]-1, self._texture.shape[1]-1
        u = int(min(w, max(0, x * w)))
        v = int(min(h, max(0, y * h)))
        return self._texture[u, v] * 255
