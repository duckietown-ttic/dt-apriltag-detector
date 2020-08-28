import os
import numpy as np
from matplotlib.pyplot import imread

from .Rectangle3 import Rectangle3


class TexturedRectangle3(Rectangle3):

    def __init__(self, name, texture, dimensions, xyz=None, rpy=None):
        Rectangle3.__init__(self, name, dimensions, xyz, rpy)
        self._texture_file = texture
        if not os.path.exists(self._texture_file) or not os.path.isfile(self._texture_file):
            raise ValueError('Could not load texture file "%s"' % self._texture_file)
        self._texture = imread(self._texture_file)[:, :, 0:3].transpose((1, 0, 2))

    def _get_uv_mapping(self, x, y, *_):
        w, h = self._texture.shape[0]-1, self._texture.shape[1]-1
        u = int(min(w, max(0, x * w)))
        v = int(min(h, max(0, y * h)))
        return self._texture[u, v] * 255

    def __iter__(self):
        for (i, j), p in self.__enumerated_iter__():
            rgb = self._get_uv_mapping(i, j)
            yield rgb, p
