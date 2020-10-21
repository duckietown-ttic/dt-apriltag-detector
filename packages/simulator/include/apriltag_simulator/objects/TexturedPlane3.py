import os
import numpy as np
from imageio import imread

from .Rectangle3 import Rectangle3
from .utils import isect_line_plane_v3


class TexturedPlane3(Rectangle3):

    def __init__(self, name, texture, xyz=None, rpy=None):
        Rectangle3.__init__(self, name, None, 0, xyz, rpy)
        if isinstance(texture, str):
            _texture_file = texture
            if not os.path.exists(_texture_file) or not os.path.isfile(_texture_file):
                raise ValueError('Could not load texture file "%s"' % _texture_file)
            self._texture = imread(_texture_file)[:, :, 0:3].transpose((1, 0, 2)) * 255
        elif callable(texture):
            self._texture = texture

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
        intersection_w = \
            isect_line_plane_v3(camera_center, ray3w, self._plane_center, self._plane_normal)
        intersection = self.transform_from_world(intersection_w)
        color3 = self._get_uv_mapping(intersection[0], intersection[1])
        return intersection_w, color3

    def _get_uv_mapping(self, x, y, *_):
        if callable(self._texture):
            return np.array(self._texture(x, y)) * 255
        elif isinstance(self._texture, np.ndarray):
            w, h = self._texture.shape[0]-1, self._texture.shape[1]-1
            u = int(min(w, max(0, x * w)))
            v = int(min(h, max(0, y * h)))
            return self._texture[u, v] * 255
