import math
from itertools import product
from multiprocessing.pool import Pool
from threading import Thread

import numpy as np

from skimage.morphology import flood_fill

from apriltag_simulator.utils import ProgressBar
from apriltag_simulator.exceptions import ObjectOutOfXBounds, ObjectOutOfYBounds
from apriltag_simulator.utils import transformations

INF = 9999999999
np.set_printoptions(suppress=True)


class Camera(object):

    @staticmethod
    def from_camera_info(name, camera_info):
        FX, _, CX, _, FY, CY, _, _, _ = camera_info['camera_matrix']['data']
        IMG_W = camera_info['image_width']
        IMG_H = camera_info['image_height']
        return Camera(name, FX, FY, CX, CY, IMG_W, IMG_H)

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
        self._Rt = np.array([
            [1,    0,    0,    0],
            [0,    1,    0,    0],
            [0,    0,    1,    0]
        ])
        self._width = width
        self._height = height
        self._lens = None
        self._CRt = np.matmul(self._C, self._Rt)

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def fx(self):
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def C(self):
        return self._C

    @property
    def lens(self):
        return self._lens

    def attach_lens(self, lens):
        from apriltag_simulator.lenses import CameraLens
        if not isinstance(lens, CameraLens):
            raise ValueError(
                'Object `lens` must be of type CameraLens, got %s instead' % str(type(lens)))
        self._lens = lens

    def detach_lens(self):
        self._lens = None

    # def capture(self, point3, no_lens=False):
    #     if isinstance(point3, np.ndarray):
    #         point3 = point3.tolist()
    #     point3 = np.array(point3 + [1])
    #
    #
    #     # apply lens distortion
    #     if self._lens is not None and not no_lens:
    #         if point3[2] == 0:
    #             return None, None
    #         xn = point3[0:2] / point3[2]
    #         # print(f'{np.round(point3, 4)} -N-> {np.round(xn, 4)}')
    #         dpoint2 = self._lens.transform(xn)
    #         # print(f'{np.round(xn, 4)} -D-> {np.round(dpoint2, 4)}')
    #         point2 = np.matmul(self._C, dpoint2.tolist() + [1])
    #         # print(f'{np.round(dpoint2, 4)} -P-> {np.round(point2, 4)}')
    #         point2 = point2[0:2]
    #     else:
    #         point2 = np.matmul(self._CRt, point3)
    #         if point2[2] == 0:
    #             return None, None
    #         point2 = point2[0:2] / point2[2]
    #     return point2
    #
    #
    #
    #
    #
    #     # point2 = np.matmul(self._CRt, point3)
    #     # if point2[2] == 0:
    #     #     return None, None
    #     # point2 = point2[0:2] / point2[2]
    #     #
    #     # point2 = np.array([(point2[0] - self._cx) / self._fx, (point2[1] - self._cy) / self._fy])
    #     #
    #     # # apply lens distortion
    #     # dpoint2 = self._lens.transform(point2)
    #     #
    #     # dpoint2 = [dpoint2[0] * self._fx + self._cx, dpoint2[1] * self._fy + self._cy]
    #     #
    #     # # print(f'{point2} -D-> {dpoint2}')
    #     #
    #     # point2 = dpoint2
    #
    #
    #     # ---
    #     return point2
    #
    # def render(self, scene, bgcolor=None, scream=False, progress=False, refine=True, return_stats=False):
    #     if bgcolor is None:
    #         bgcolor = 0
    #
    #     img = np.full((self._width, self._height, 3), -1)
    #     stats = {
    #         'min_distortion_px': INF,
    #         'max_distortion_px': -INF,
    #         'horizontal_fov_deg': 0,
    #         'vertical_fov_deg': 0,
    #         'diagonal_fov_deg': 0,
    #         'rectified_uv_domain': {
    #             'u': [INF, -INF],
    #             'v': [INF, -INF]
    #         }
    #     }
    #
    #     for obj in scene:
    #         pbar = ProgressBar(header='Object: %s' % obj.name)
    #         num_points = obj.num_points()
    #         # iterate over the object points
    #         i = 0
    #         for point3, color3 in obj.points():
    #             u, v = self.capture(point3)
    #
    #             # print(f'[{np.round(point3, 3)}] -> [{u}, {v}]')
    #
    #             if u is None:
    #                 continue
    #             u, v = np.array([u, v]).round().astype(np.int32).tolist()
    #
    #             if progress:
    #                 pbar.update(100 * i / num_points)
    #             if 0 <= u < self._width and 0 <= v < self._height:
    #                 img[u, v] = color3
    #
    #                 if return_stats:
    #                     # compute minimum/maximum distortion
    #                     u1, v1 = self.capture(point3, no_lens=True)
    #                     if u1 is None:
    #                         continue
    #                     u1, v1 = np.array([u1, v1]).round().astype(np.int32).tolist()
    #
    #                     distortion = np.linalg.norm([u - u1, v - v1])
    #                     stats['min_distortion_px'] = min(stats['min_distortion_px'], distortion)
    #                     stats['max_distortion_px'] = max(stats['max_distortion_px'], distortion)
    #                     # compute effective FoV
    #                     stats['horizontal_fov_deg'] = max(
    #                         stats['horizontal_fov_deg'],
    #                         2 * np.rad2deg(np.arctan(point3[0] / point3[2]))
    #                     )
    #                     stats['vertical_fov_deg'] = max(
    #                         stats['vertical_fov_deg'],
    #                         2 * np.rad2deg(np.arctan(point3[1] / point3[2]))
    #                     )
    #                     stats['diagonal_fov_deg'] = np.linalg.norm(
    #                         [stats['horizontal_fov_deg'], stats['vertical_fov_deg']])
    #                     # compute domains for u, v in rectified image
    #                     stats['rectified_uv_domain']['u'][0] = min(
    #                         stats['rectified_uv_domain']['u'][0], u1)
    #                     stats['rectified_uv_domain']['u'][1] = max(
    #                         stats['rectified_uv_domain']['u'][1], u1)
    #                     stats['rectified_uv_domain']['v'][0] = min(
    #                         stats['rectified_uv_domain']['v'][0], v1)
    #                     stats['rectified_uv_domain']['v'][1] = max(
    #                         stats['rectified_uv_domain']['v'][1], v1)
    #             else:
    #                 if scream and (u < 0 or u >= self._width) and np.sum(color3) == 0:
    #                     if progress:
    #                         pbar.done()
    #                     raise ObjectOutOfXBounds()
    #                 if scream and (v < 0 or v >= self._height) and np.sum(color3) == 0:
    #                     if progress:
    #                         pbar.done()
    #                     raise ObjectOutOfYBounds()
    #
    #             i += 1
    #
    #         if progress:
    #             pbar.done()
    #
    #     if refine:
    #         # fill background pixels
    #         corners = [(0, 0), (self._width-1, 0), (self._width-1, self._height-1), (0, self._height-1)]
    #         selem = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    #         for u, v in corners:
    #             if img[u, v, 0] == -1:
    #                 img[:, :, 0] = flood_fill(img[:, :, 0], (u, v), bgcolor, selem=selem)
    #                 img[:, :, 1] = flood_fill(img[:, :, 1], (u, v), bgcolor, selem=selem)
    #                 img[:, :, 2] = flood_fill(img[:, :, 2], (u, v), bgcolor, selem=selem)
    #
    #         # find all -1 pixels
    #         neigh_mask = [
    #             [-1, -1], [0, -1], [1, -1],
    #             [-1,  0],          [1,  0],
    #             [-1,  1], [0,  1], [1,  1]
    #         ]
    #         empty_pixels = np.argwhere(img == -1)[:, 0:2]
    #         indices = np.unique(empty_pixels, axis=0) if len(empty_pixels) else []
    #
    #         # fill -1 pixels with the most common color among the neighboring ones
    #         for pixel in indices:
    #             neighs = np.add(neigh_mask, pixel)
    #             neighs = [n for n in neighs if 0 <= n[0] < self._width and 0 <= n[1] < self._height]
    #             colors = img[tuple(np.array(neighs).T)]
    #             colors, count = np.unique(colors, axis=0, return_counts=True)
    #             color = colors[np.argmax(count)]
    #             img[pixel[0], pixel[1]] = color
    #     # ---
    #     img = img.astype(np.uint8)
    #     if return_stats:
    #         return img, stats
    #     return img

    def render(self, scene, bgcolor=None, num_threads=4, return_stats=False, progress=False):
        img = np.full((self._width, self._height, 3), bgcolor)
        # split rendering job to num_threads workers
        pool = Pool(num_threads)
        args = []
        for i in range(num_threads):
            args.append((i, num_threads, self, scene))
        # spin workers
        res = pool.starmap(ray_casting_task, args)
        # combine results
        for ma in res:
            img = np.ma.where(ma == -1, img, ma)
        # ---
        return img


def ray_casting_task(cur, tot, camera, scene):
    split = 1 / tot
    vs = range(int(cur * split * camera.height), int((cur + 1) * split * camera.height), 1)
    img = np.full((camera.width, camera.height, 3), -1)
    invM = transformations.inverse_matrix(camera.C)
    # shoot rays
    for _u in range(camera.width):
        for _v in vs:
            u, v = camera.lens.to_pinhole_pixel(_u, _v)
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and (z is None or inters_w[2] < z):
                    z, c = inters_w[2], color
            if c is not None:
                img[_u, _v] = c
    # ---
    return img
