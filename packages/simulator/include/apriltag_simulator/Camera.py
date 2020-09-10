import numpy as np
from multiprocessing.pool import Pool

from apriltag_simulator.constants import NUM_THREADS
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
    def fx(self):
        return self._fx

    @property
    def fy(self):
        return self._fy

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def aspect_ratio(self):
        return self._width / self._height

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

    def render(self, scene, bgcolor=None, num_threads: int = NUM_THREADS):
        img = np.full((self._width, self._height, 3), bgcolor)
        # split rendering job into num_threads workers
        args = []
        for i in range(num_threads):
            args.append((i, num_threads, self, scene))
        # spin workers
        with Pool(num_threads) as pool:
            res = pool.starmap(ray_casting_task, args)
        # combine results
        for ma in res:
            img = np.ma.where(ma == -1, img, ma)
        # ---
        return img.astype(np.uint8)

    def to_json(self):
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height
        }


def ray_casting_task(cur, tot, camera, scene):
    split = 1 / tot
    vs = range(int(cur * split * camera.height), int((cur + 1) * split * camera.height), 1)
    img = np.full((camera.width, camera.height, 3), -1)
    invM = transformations.inverse_matrix(camera.C)
    # shoot rays
    for _u in range(camera.width):
        for _v in vs:
            u, v = camera.lens.to_pinhole_pixel(_u, _v)
            if u is None or v is None:
                # we don't have a mapping between distorted pixel and underlying pinhole pixel
                continue
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
