import math
import time
import numpy as np
from itertools import product, chain
from multiprocessing.pool import Pool

from .CameraLens import CameraLens
from apriltag_simulator.Camera import Camera
from apriltag_simulator.constants import NUM_THREADS, INF


class FishEyeLens(CameraLens):

    @staticmethod
    def from_camera_info(name, camera, camera_info):
        K1, K2, P1, P2, K3 = camera_info['distortion_coefficients']['data']
        return FishEyeLens(name, camera, K1, K2, K3, P1, P2)

    def __init__(self, name: str, camera: Camera,
                 k1: float = 0, k2: float = 0, k3: float = 0, p1: float = 0, p2: float = 0,
                 generate_inverse_map: bool = True):
        CameraLens.__init__(self, name)
        self._name = name
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2
        self._camera = camera
        self._map = None
        self._ph_camera = None
        if generate_inverse_map:
            self._map, self._ph_camera = self._create_map()

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def p1(self):
        return self._p1

    @property
    def p2(self):
        return self._p2

    @property
    def k3(self):
        return self._k3

    @property
    def underlying_pinhole_camera(self) -> Camera:
        return self._ph_camera

    @property
    def map(self) -> np.ndarray:
        return self._map

    def rectify(self, u: int, v: int) -> np.array:
        _u, _v = self._map[u, v]
        return (_u, _v) if (_u != INF and _v != INF) else (None, None)

    def maximum_distortion(self):
        m = 0
        for u, v in product(range(self._camera.width), range(self._camera.height)):
            _u, _v = self.rectify(u, v)
            if _u is None or _v is None:
                continue
            _m = np.linalg.norm([u - _u, v - _v])
            m = max(m, _m)
        return int(m)

    def transform(self, x: float, y: float) -> np.array:
        p = np.array([x, y])
        r = np.linalg.norm(p)
        radial = (1 + self._k1 * (r ** 2) + self._k2 * (r ** 4) + self._k3 * (r ** 6)) * p
        tangential = [
            2 * self._p1 * x * y + self._p2 * ((r ** 2) + 2 * (x ** 2)),
            2 * self._p2 * x * y + self._p1 * ((r ** 2) + 2 * (y ** 2))
        ]
        return np.add(radial, tangential)

    def distort(self, u: int, v: int, cx=None, cy=None, fx=None, fy=None):
        cx = self._camera.cx if cx is None else cx
        cy = self._camera.cy if cy is None else cy
        fx = self._camera.fx if fx is None else fx
        fy = self._camera.fy if fy is None else fy
        # ---
        i = (u - cx) / fx
        j = (v - cy) / fy
        i, j = self.transform(i, j)
        _u = int(math.floor(i * self._camera.fx + self._camera.cx))
        _v = int(math.floor(j * self._camera.fy + self._camera.cy))
        if 0 <= _u < self._camera.width and 0 <= _v < self._camera.height:
            return _u, _v
        return None, None

    def _create_map(self, num_threads: int = NUM_THREADS) -> (np.array, Camera):
        stime = time.time()
        lens = np.full((self._camera.width, self._camera.height, 2), INF)
        # find underlying pinhole cameras for inner and outer frame
        (innerW, innerH), (outerW, outerH) = self.find_underlying_pinhole_camera()
        # stats
        print(f'Lens[{self._name}]underlying-cameras: '
              f'Inner [res: {innerW}x{innerH}];  '
              f'Outer [res: {outerW}x{outerH}]')
        # use outer frame so to capture a full image of distorted pixels w/ original camera params
        width, height = outerW, outerH
        # scale image center
        ncx = width * (self._camera.cx / self._camera.width)
        ncy = height * (self._camera.cy / self._camera.height)
        # create underlying pinhole camera
        ph_camera = Camera(
            f'lens[{self._name}]induced-pinhole-camera',
            self._camera.fx, self._camera.fy, ncx, ncy, width, height
        )
        # profiling
        print(f'Underlying pinhole camera parameters found in {int(time.time() - stime)} secs')
        stime = time.time()

        # split vectors generation job into num_threads workers
        args = []
        for i in range(num_threads):
            args.append((i, num_threads, self, self._camera, ph_camera))
        # spin workers
        with Pool(num_threads) as pool:
            res = pool.starmap(lens_map_generation_task, args)
        # combine results
        for ma in res:
            lens = np.ma.where(ma == INF, lens, ma)
        # profiling
        print(f'Lens map generated in {int(time.time() - stime)} secs')
        stime = time.time()

        # fill holes in lens map
        mask_size = 2
        lensc = np.full((self._camera.width, self._camera.height, 2), INF)
        # create a mask of size [mask_size x mask_size]
        mask_range = list(range(-mask_size, mask_size + 1, 1))
        neigh_mask = list(product(mask_range, mask_range))
        # find pixels in map with no corresponding rectified pixel
        empty_pixels = np.argwhere(lens == INF)[:, 0:2]
        indices = np.unique(empty_pixels, axis=0) if len(empty_pixels) else []
        # use average from neighbors as value for each empty pixel
        for px in indices:
            neighs = np.add(px, neigh_mask)
            neighs = [
                lens[n[0], n[1]] for n in neighs
                if 0 <= n[0] < self._camera.width and 0 <= n[1] < self._camera.height
                and lens[n[0], n[1], 0] != INF
            ]
            lensc[px[0], px[1]] = np.floor(np.mean(neighs, axis=0)).astype(int)
        lens = np.ma.where(lens == INF, lensc, lens)
        # profiling
        print(f'Lens map filling completed in {int(time.time() - stime)} secs')
        # ---
        return lens.astype(int), ph_camera

    def find_underlying_pinhole_camera(self):
        # find inner and outer contours
        innerW, innerH, outerW, outerH = None, None, None, None
        # find inner then outer frame
        for foi in ['inner', 'outer']:
            step = 100
            direction = 1
            cursor = 2
            search_space = [0, INF]
            # grow frame step-pixels at a time
            while True:
                w, h = cursor, int(math.floor(cursor / self._camera.aspect_ratio))
                # compute corners
                w2 = int(math.floor(w * 0.5))
                h2 = int(math.floor(h * 0.5))
                # create frame of pixels
                frame = chain(
                    # top border
                    product(range(-w2, w2, 1), [-h2]),
                    # left border
                    product([-w2], range(-h2 + 1, h2 - 1, 1)),
                    # right border
                    product([w2], range(-h2 + 1, h2 - 1, 1)),
                    # bottom border
                    product(range(-w2, w2, 1), [h2])
                )
                # distort frame (px-by-px) and check if the distorted pixels fall within the image
                hits = 0
                frame_size = 0
                for p in frame:
                    frame_size += 1
                    px = np.add(p, [self._camera.cx, self._camera.cy]).astype(int)
                    u, v = self.distort(*px)
                    if u is None or v is None:
                        continue
                    if not (0 <= u < self._camera.width and 0 <= v < self._camera.height):
                        continue
                    # ---
                    hits += 1

                bstep = int(math.ceil((search_space[1] - search_space[0]) * 0.5))
                if bstep <= 1:
                    # nothing left to search
                    break

                if foi == 'outer':
                    if hits == 0:
                        # outer frame candidate
                        outerW = w
                        outerH = h
                        # this is an outer frame, this is an upper bound for outer frame
                        search_space[1] = cursor
                        direction = -1
                    else:
                        # this frame has hits, it is a lower bound for outer frame
                        search_space[0] = cursor
                        direction = 1

                if foi == 'inner':
                    if hits == frame_size:
                        # inner frame candidate
                        innerW = w
                        innerH = h
                        # this frame has all hits, it is a lower bound for inner frame
                        search_space[0] = cursor
                        direction = 1
                    else:
                        # this frame has partial to no hits, it is an upper bound for inner frame
                        search_space[1] = cursor
                        direction = -1

                # update cursor
                if search_space[1] == INF:
                    # no upper bound yet, constant step
                    cursor += step
                else:
                    # there is an upper bound, perform binary search
                    bstep = max(int(math.floor((search_space[1] - search_space[0]) * 0.5)), 1)
                    cursor += direction * bstep
                    assert bstep > 0
        # ---
        return (innerW, innerH), (outerW, outerH)


def lens_map_generation_task(cur, tot, lens, camera, ph_camera):
    split = 1 / tot
    vs = range(int(cur * split * ph_camera.height), int((cur + 1) * split * ph_camera.height), 1)
    lens_map = np.full((camera.width, camera.height, 2), INF)
    # shoot rays
    for u in range(ph_camera.width):
        for v in vs:
            # find where u, v will appear in the distorted image of the physical camera
            _u, _v = lens.distort(u, v, cx=ph_camera.cx, cy=ph_camera.cy)
            if _u is None or _v is None:
                # it is a miss
                continue
            # convert from the reference frame of the underlying camera to the physical camera's
            u1 = u - ph_camera.cx + camera.cx
            v1 = v - ph_camera.cy + camera.cy
            # ---
            if 0 <= _u < camera.width and 0 <= _v < camera.height:
                lens_map[_u, _v] = [u1, v1]
    # ---
    return lens_map
