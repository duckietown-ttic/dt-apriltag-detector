import os
import numpy as np

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')

debug_factor = .2

# wanted resolution
Rw = 2592
Rh = 1944
F = Rw

# NOTE: from the Apriltag paper
# Rw = 400
# Rh = 400

# maximum distortion (one pixel can be moved by at most 60% of Rw)
Md = 0.6

# compute effective resolution (including a frame to allow for distortion)
r_w = int(Rw * (1 + Md))
r_h = int(Rh * (1 + Md))
camera_elevation = 3

# define focal length
f = int(F * (1 + Md))

# apply debug factor
r_w = int(r_w * debug_factor)
r_h = int(r_h * debug_factor)
f = f * debug_factor

# define checkerboard floor
grid_resolution = 1000
grid_ratio = 8 / 3
grid_xyz = [0, camera_elevation, 0]
grid_rpw = [-np.deg2rad(90), 0, 0]
grid_filepath = f'{DATA_DIR}/simulator/background.png'

# define infinite generator for Z
# NOTE: computed from step 5
_z_step = 15
_max_Z = 100
_y_num_locations = 3
_x_num_locations = 4

Z = [1.5, 2.5, 5, 10, 15, 20] + list(range(20 + _z_step, _max_Z, _z_step))


def X(_z):
    _m = (r_w / 2) / f
    _max_w = _m - tag_size / (_z if _z > 0 else 1)
    _values = np.linspace(0, _max_w * _z, _x_num_locations).round(3)
    _values = np.maximum(_values, [0])
    _values = sorted(set(_values.tolist()))
    return _values


def Y(_z):
    _m = (r_h / 2) / f
    _max_h = _m - tag_size / (_z if _z > 0 else 1)
    _values = np.linspace(0, _max_h * _z, _y_num_locations).round(3)
    _values = -np.maximum(_values, [0])
    _values = sorted(set(_values.tolist()), reverse=True)
    return _values


# define domains for roll, pitch, and yaw
roll = [0, 30, 60]
pitch = [0, 30, 60]
yaw = [0, 30]

# define domains for distortion parameters
# NOTE: computed from steps 1-3
k1 = [-0.4, -0.25, -0.05, 0, 0.02, 0.05, 0.15]
k2 = lambda _k1: [_k1 * 0.019 + 0.805 * (_k1 ** 2)]
p1 = [0, 0.01]
p2 = [0, 0.01]
k3 = [0]

# define apriltag size and tag info
tag_size = 1
tag_ratio = 0.8
tag_w = tag_size * tag_ratio
tag_h = tag_size * tag_ratio
tag_id = 0
tag_texture = f'{DATA_DIR}/simulator/tags/tag{tag_id}.png'


def grid_texture(_x, _y):
    _sx, _sy = np.sign(_x) == 1, np.sign(_y) == 1
    _x, _y = int(abs(_x)), int(abs(_y))
    _i, _j = int(_x % grid_resolution), int(_y % grid_resolution)
    _intensity = 0 if (int(_i + _j + int(_sx) + int(_sy)) % 2 == 0) else 1
    return [_intensity] * 3


# define pinhole camera
pinhole_camera_info = {
    "image_width": r_w,
    "image_height": r_h,
    "camera_matrix": {
        "data": [
            f, 0.0, r_w / 2,
            0.0, f, r_h / 2,
            0.0, 0.0, 1.0
        ]
    }
}


def generate_camera_info(_scale=1.0):
    return {
        "image_width": int(r_w * _scale),
        "image_height": int(r_h * _scale),
        "camera_matrix": {
            "data": [
                f * _scale, 0.0, r_w * 0.5 * _scale,
                0.0, f * _scale, r_h * 0.5 * _scale,
                0.0, 0.0, 1.0
            ]
        }
    }


# export constants
__all__ = [
    'DATA_DIR',
    'r_w', 'r_h',
    'tag_size', 'tag_w', 'tag_h', 'tag_id', 'tag_texture',
    'grid_texture', 'grid_xyz', 'grid_rpw', 'grid_filepath',
    'roll', 'pitch', 'yaw',
    'k1', 'k2', 'p1', 'p2', 'k3',
    'Z', 'X', 'Y',
    'pinhole_camera_info', 'generate_camera_info'
]
