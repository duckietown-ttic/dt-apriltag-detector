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
            res = pool.starmap(ray_casting_through_lens_task, args)
        # combine results
        for ma in res:
            img = np.ma.where(ma == INF, img, ma)
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


def ray_casting_through_lens_task(cur, tot, camera, scene):
    split = 1 / tot
    vs = range(int(cur * split * camera.height), int((cur + 1) * split * camera.height), 1)
    img = np.full((camera.width, camera.height, 3), INF)
    invM = transformations.inverse_matrix(camera.C)
    # shoot rays
    for _u in range(camera.width):
        for _v in vs:
            # find the pixel's location on the underlying pinhole camera image
            u, v = camera.lens.to_pinhole_pixel(_u, _v)
            if u is None or v is None:
                # we don't have a mapping between distorted pixel and underlying pinhole pixel
                continue
            # find the ray going through the pixel u, v
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            # find the closest intersection (if any)
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and (z is None or inters_w[2] < z):
                    z, c = inters_w[2], color
            # check if it hit anything
            if c is None:
                # the ray is a miss
                continue
            # ---
            img[_u, _v] = c
    # ---
    return img


def ray_casting_through_underlying_pinhole_task(cur, tot, camera, scene):
    split = 1 / tot
    ph_camera = camera.lens.underlying_pinhole_camera
    vs = range(int(cur * split * ph_camera.height), int((cur + 1) * split * ph_camera.height), 1)
    img = np.full((camera.width, camera.height, 3), INF)
    invM = transformations.inverse_matrix(ph_camera.C)
    # shoot rays
    for u in range(ph_camera.width):
        for v in vs:
            # find the ray going through the pixel u, v
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            # find the closest intersection (if any)
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and (z is None or inters_w[2] < z):
                    z, c = inters_w[2], color
            # check if it hit anything
            if c is None:
                # the ray is a miss
                continue
            # compute where the u, v point will appear in the distorted image
            _u, _v = camera.lens.distort(
                u, v, ph_camera.cx, ph_camera.cy, ph_camera.fx, ph_camera.fy)
            if _u is None or _v is None:
                # we don't have a mapping between distorted pixel and underlying pinhole pixel
                continue
            # ray was a hit and the point is visible inside the distorted image
            img[_u, _v] = c
    # ---
    return img
