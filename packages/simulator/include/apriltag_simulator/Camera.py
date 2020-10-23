import numpy as np
from multiprocessing.pool import Pool

from apriltag_simulator.constants import NUM_THREADS, INF
from apriltag_simulator.utils import transformations
from apriltag_simulator.constants import ROI

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

    def render(self, scene, bgcolor=None, num_threads: int = NUM_THREADS, roi: ROI = None):
        img = np.full((self._width, self._height, 3), bgcolor)

        # find bounding box for the scene content
        mx, my, Mx, My = self.width, self.height, 0, 0
        for obj in scene.objects():
            for p in obj.shadow_polygon():
                u, v = self._render_point3(*p)
                if u is None or v is None:
                    continue
                mx = int(np.floor(max(min(mx, u), 0)))
                my = int(np.floor(max(min(my, v), 0)))
                Mx = int(np.ceil(max(max(Mx, u), self.width)))
                My = int(np.ceil(max(max(My, v), self.height)))
        roi = ROI(x=mx, y=my, width=Mx - mx, height=My - my)

        # split rendering job into num_threads workers
        args = []
        for i in range(num_threads):
            args.append((i, num_threads, self, scene, roi))
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

    def _render_point3(self, x, y, z):
        p2 = [x / z if z else 0, y / z if z else 0, 1]
        u, v = np.dot(self.C, p2)[:2]
        if self.lens:
            # find the pixel's location on the underlying pinhole camera image
            u, v = self.lens.rectify(u, v)
            if u is None or v is None:
                # we don't have a mapping between distorted pixel and underlying pinhole pixel
                return None, None
        return u, v


def ray_casting_through_lens_task(cur, tot, camera, scene, roi: ROI = None):
    if roi is None:
        roi = ROI(x=0, y=0, width=camera.width, height=camera.height)
    split = 1 / tot
    us = range(roi.x, camera.width, 1)
    vs = range(roi.y + int(cur * split * roi.height),
               roi.y + int((cur + 1) * split * roi.height), 1)
    img = np.full((camera.width, camera.height, 3), INF)
    invM = transformations.inverse_matrix(camera.C)
    # shoot rays
    for _u in us:
        for _v in vs:
            if camera.lens:
                # find the pixel's location on the underlying pinhole camera image
                u, v = camera.lens.rectify(_u, _v)
                if u is None or v is None:
                    # we don't have a mapping between distorted pixel and underlying pinhole pixel
                    continue
            else:
                # this is a pinhole camera, no lens attached
                u, v = _u, _v
            # find the ray going through the pixel u, v
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            # find the closest intersection (if any)
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and inters_w[2] > 0 and (z is None or inters_w[2] < z):
                    z, c = inters_w[2], color
            # check if it hit anything
            if c is None:
                # the ray is a miss
                continue
            # ---
            if 0 <= _u < camera.width and 0 <= _v < camera.height:
                img[_u, _v] = c
    # ---
    return img


def ray_casting_through_lens_task2(cur, tot, camera, scene):
    split = 1 / tot
    vs = range(int(cur * split * camera.height), int((cur + 1) * split * camera.height), 1)
    img = np.full((camera.width, camera.height, 3), INF)
    invM = transformations.inverse_matrix(camera.C)

    k1 = camera.lens.k1
    k2 = camera.lens.k2
    p1 = camera.lens.p1
    p2 = camera.lens.p2
    k3 = camera.lens.k3
    print(f'D: [{k1}, {k2}, {p1}, {p2}, {k3}]')
    uc, vc = camera.lens.distort(camera.cx, camera.cy)
    xc, yc = (uc - camera.cx) / camera.fx, (vc - camera.cy) / camera.fy
    print(f'cd = [{uc}, {vc}]')
    print(f'cxd = [{xc}, {yc}]')


    # shoot rays
    for _u in range(camera.width):
        for _v in vs:


            # # find the pixel's location on the underlying pinhole camera image
            # u, v = camera.lens.rectify(_u, _v)
            # if u is None or v is None:
            #     # we don't have a mapping between distorted pixel and underlying pinhole pixel
            #     continue


            xd, yd = (_u - camera.cx) / camera.fx, (_v - camera.cy) / camera.fy

            r = np.sqrt((xd-xc)**2 + (yd-yc)**2)
            xu = xd + (xd - xc)*(k1 * r**2 + k2 * r**4 + k3 * r**6) # + (p1 * (r**2 + 2*(xd-xc)**2) + 2*p2*(xd-xc)*(yd-yc))
            yu = yd + (yd - yc)*(k1 * r**2 + k2 * r**4 + k3 * r**6) # + (p2 * (r**2 + 2*(yd-yc)**2) + 2*p1*(xd-xc)*(yd-yc))

            u = int(xu * camera.fx + camera.cx)
            v = int(yu * camera.fy + camera.cy)


            # find the ray going through the pixel u, v
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            # find the closest intersection (if any)
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and inters_w[2] > 0 and (z is None or inters_w[2] < z):
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
                if inters_w and inters_w[2] > 0 and (z is None or inters_w[2] < z):
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


def ray_casting_through_underlying_pinhole_double_lens_task(cur, tot, camera, scene, ilens):
    split = 1 / tot
    ph_camera = camera.lens.underlying_pinhole_camera
    vs = range(int(cur * split * ph_camera.height), int((cur + 1) * split * ph_camera.height), 1)
    img = np.full((camera.width, camera.height, 3), INF)
    invM = transformations.inverse_matrix(ph_camera.C)

    iph_camera = ilens.underlying_pinhole_camera

    # shoot rays
    for u in range(ph_camera.width):
        for v in vs:
            # find the ray going through the pixel u, v
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            # find the closest intersection (if any)
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and inters_w[2] > 0 and (z is None or inters_w[2] < z):
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

            print(f'{_u}, {_v}  ->  ', end='')
            _u1, _v1 = ilens.distort(_u, _v)
            print(f'{_u1}, {_v1}')
            if _u1 is None or _v1 is None:
                # we don't have a mapping between distorted pixel and underlying pinhole pixel
                continue

            img[_u1, _v1] = c

            # ray was a hit and the point is visible inside the distorted image
            # img[_u, _v] = c
    # ---
    return img



def ray_casting_through_underlying_pinhole_w_supersampling_task(cur, tot, camera, scene, supersampling=2.0):
    split = 1 / tot
    ph_camera = camera.lens.underlying_pinhole_camera
    vs = range(int(cur * split * ph_camera.height * supersampling), int((cur + 1) * split * ph_camera.height * supersampling), 1)
    img = np.full((camera.width, camera.height, 3), INF)
    invM = transformations.inverse_matrix(ph_camera.C)

    # shoot rays
    for u1 in range(int(ph_camera.width * supersampling)):
        for v1 in vs:
            u, v = u1 / supersampling, v1 / supersampling
            # find the ray going through the pixel u, v
            ray = np.matmul(invM, [u, v, 1])
            z, c = None, None
            # find the closest intersection (if any)
            for obj in scene:
                inters_w, color = obj.intersect(ray)
                if inters_w and inters_w[2] > 0 and (z is None or inters_w[2] < z):
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
