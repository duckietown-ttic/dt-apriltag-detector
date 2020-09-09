import math
import time
from itertools import product, chain
import matplotlib.pyplot as plt

from .CameraLens import CameraLens
from apriltag_simulator.Camera import Camera

import numpy as np


class FishEyeLens(CameraLens):

    @staticmethod
    def from_camera_info(name, camera, camera_info):
        K1, K2, P1, P2, K3 = camera_info['distortion_coefficients']['data']
        return FishEyeLens(name, camera, K1, K2, K3, P1, P2)

    def __init__(self, name, camera, k1=0, k2=0, k3=0, p1=0, p2=0):
        CameraLens.__init__(self, name)
        self._name = name
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2
        self._camera = camera
        self._map = self._create_map()

    def to_pinhole_pixel(self, u, v):
        return self._map[u, v]

    def transform(self, point2):
        if not isinstance(point2, np.ndarray):
            point2 = np.array(point2)
        x, y = point2.tolist()
        r = np.linalg.norm(point2)
        radial = (1 + self._k1 * (r ** 2) + self._k2 * (r ** 4) + self._k3 * (r ** 6)) * point2
        tangential = np.array([
            2 * self._p1 * x * y + self._p2 * ((r ** 2) + 2 * (x ** 2)),
            2 * self._p2 * x * y + self._p1 * ((r ** 2) + 2 * (y ** 2))
        ])
        return np.add(radial, tangential)

    def _distort(self, u, v):
        i = (u - self._camera.cx) / self._camera.fx
        j = (v - self._camera.cy) / self._camera.fy
        i, j = self.transform([i, j])
        _u = int(math.floor(i * self._camera.fx + self._camera.cx))
        _v = int(math.floor(j * self._camera.fy + self._camera.cy))
        if 0 <= _u < self._camera.width and 0 <= _v < self._camera.height:
            return _u, _v
        return None, None

    def _create_map(self):
        lens = np.full((self._camera.width, self._camera.height, 2), -1)
        lensc = np.full((self._camera.width, self._camera.height, 2), -1)


        # img = np.full((self._camera.width, self._camera.height, 3), 255)
        # plot = plt.imshow(img.transpose((1, 0, 2)))
        # plt.show(block=False)


        # find inner and outer contours
        i = 2
        stop = False
        ratio = self._camera.height / self._camera.width
        innerW, innerH, outerW, outerH = None, None, None, None
        minX, maxX, minY, maxY = None, None, None, None

        while not stop:
            w, h = i, int(math.floor(i * ratio))
            # compute corners
            w2 = int(math.floor(w * 0.5))
            h2 = int(math.floor(h * 0.5))

            frame = chain(
                # top line
                product(range(-w2, w2, 1), [-h2]),
                # left line
                product([-w2], range(-h2 + 1, h2 - 1, 1)),
                # right line
                product([w2], range(-h2 + 1, h2 - 1, 1)),
                # bottom line
                product(range(-w2, w2, 1), [h2])
            )

            img = np.full((self._camera.width, self._camera.height, 3), 255)

            hits = 0
            frame_size = 0
            for p in frame:
                frame_size += 1
                px = np.add(p, [self._camera.width * 0.5, self._camera.height * 0.5]).astype(int)
                u, v = self._distort(*px)
                if u is None or v is None:
                    continue
                if not (0 <= u < self._camera.width and 0 <= v < self._camera.height):
                    continue
                # ---
                hits += 1

                img[u, v, :] = 0
                if 0 <= px[0] < self._camera.width and 0 <= px[1] < self._camera.height:
                    img[px[0], px[1]] = [255, 0, 0]

                # store min and max for X and Y
                minX = px[0] if minX is None else min(minX, px[0])
                maxX = px[0] if maxX is None else max(maxX, px[0])
                minY = px[1] if minY is None else min(minY, px[1])
                maxY = px[1] if maxY is None else max(maxY, px[1])
            # ---
            if hits > 0:
                # outer frame candidate
                outerW = maxX - minX
                outerH = maxY - minY
            if hits == frame_size:
                # inner frame candidate
                innerW = maxX - minX
                innerH = maxY - minY
            if hits == 0:
                # outer frame reached
                stop = True

            # plot.set_data(img)
            # plt.pause(0.0001)

            i += 8

        print([outerW, outerH])
        print([innerW, innerH])

        width, height = outerW, outerH

        plt.close()


        # create new temporary camera
        # # TODO: it might be that we have to leave these unscaled
        # nfx = width * (self._camera.fx / self._camera.width)
        # nfy = height * (self._camera.fy / self._camera.height)

        ncx = width * (self._camera.cx / self._camera.width)
        ncy = height * (self._camera.cy / self._camera.height)

        # print(f'{self._camera.cx}  ->  {ncx}')
        # print(f'{self._camera.cy}  ->  {ncy}')

        # ncamera = Camera('lens_camera', nfx, nfy, ncx, ncy, width, height)
        # ncamera = Camera('lens_camera', self._camera.fx, self._camera.fy, ncx, ncy, width, height)


        for u in range(width):
            for v in range(height):
                i = (u - ncx) / self._camera.fx
                j = (v - ncy) / self._camera.fy
                i, j = self.transform([i, j])


                _u = int(math.floor(i * self._camera.fx + self._camera.cx))
                _v = int(math.floor(j * self._camera.fy + self._camera.cy))

                # _u, _v = camera.lens._distort(u, v)

                u1 = u - ncx + self._camera.cx
                v1 = v - ncy + self._camera.cy

                if _u is not None and _v is not None:
                    if 0 <= _u < self._camera.width and 0 <= _v < self._camera.height:
                        lens[_u, _v] = [u1, v1]

        img = np.full((self._camera.width, self._camera.height, 3), 0)
        img[:, :, 0:2] = 255 * (lens != -1)
        plt.imshow(img.transpose((1, 0, 2)))
        plt.show()

        mask_size = 2
        mask_range = list(range(-mask_size, mask_size + 1, 1))
        neigh_mask = list(product(mask_range, mask_range))

        empty_pixels = np.argwhere(lens == -1)[:, 0:2]
        indices = np.unique(empty_pixels, axis=0) if len(empty_pixels) else []
        for px in indices:
            neighs = np.add(px, neigh_mask)
            neighs = [
                lens[n[0], n[1]] for n in neighs
                if 0 <= n[0] < self._camera.width and 0 <= n[1] < self._camera.height
                   and lens[n[0], n[1], 0] != -1
            ]
            assert len(neighs) > 0
            lensc[px[0], px[1]] = np.floor(np.mean(neighs, axis=0)).astype(int)

        # lens = np.ma.where(lens == -1, lensc, lens)
        return lens.astype(int)