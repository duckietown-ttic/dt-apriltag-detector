import os
import numpy as np

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')

debug_factor = .1

# wanted resolution
Rw = 2592
Rh = 1944
# maximum distortion (one pixel can be moved by at most 30% of Rw)
Md = 0.3

# compute effective resolution (including a frame to allow for distortion)
r_w = int(Rw * (1 + Md))
r_h = int(Rh * (1 + Md))
# define focal length
f = r_w * 0.5

# define infinite generator for Z
_Z = [.4, .6, .8, 1, 1.5, 2]
_z_step = 1


def Z():
    for _z in _Z:
        yield _z
    _z = _Z[-1] + _z_step
    while True:
        yield _z
        _z += _z_step


def X(_z, _camera):
    # TODO: increase x by a step function of z and stop when the point goes out of FoV
    pass


def Y(_z, _camera):
    # TODO: increase y by a step function of z and stop when the point goes out of FoV
    pass


# define domains for roll, pitch, and yaw
roll = [0, 30, 60]
pitch = [0, 30, 60]
yaw = [0, 30]

# define domains for distortion parameters
k1 = [-0.4, -0.25, -0.05, 0, 0.015, 0.05, 0.15]
k2 = lambda _k1: _k1 * 0.019 + 0.805 * (_k1 ** 2)
p1 = [0, 0.01]
p2 = [0, 0.01]
k3 = [0]

# define apriltag size and tag info
tag_size = 0.1
tag_ratio = 0.8
tag_w = tag_size * tag_ratio
tag_h = tag_size * tag_ratio
tag_id = 0
tag_texture = f'{DATA_DIR}/simulator/tags/tag{tag_id}.png'

# define checkerboard floor
grid_texture = f'{DATA_DIR}/simulator/misc/checkerboard.png'
grid_ratio = 8 / 3
grid_w = 200
grid_h = int(grid_w / grid_ratio)
grid_xyz = [0, 3, grid_h * 0.5]
grid_rpw = [-np.deg2rad(80), 0, 0]

# define pinhole camera
pinhole_camera_info = {
    "image_width": r_w,
    "image_height": r_h,
    "camera_matrix": {
      "data": [
          f,      0.0,    r_w/2,
          0.0,      f,    r_h/2,
          0.0,    0.0,      1.0
      ]
    }
}

# scale camera in debug mode
pinhole_camera_info["image_width"] = int(pinhole_camera_info["image_width"] * debug_factor)
pinhole_camera_info["image_height"] = int(pinhole_camera_info["image_height"] * debug_factor)
pinhole_camera_info["camera_matrix"]["data"][0] *= debug_factor
pinhole_camera_info["camera_matrix"]["data"][2] *= debug_factor
pinhole_camera_info["camera_matrix"]["data"][4] *= debug_factor
pinhole_camera_info["camera_matrix"]["data"][5] *= debug_factor

# export constants
__all__ = [
    'DATA_DIR',
    'r_w', 'r_h',
    'tag_size', 'tag_w', 'tag_h', 'tag_id',
    'grid_w', 'grid_h', 'grid_texture', 'grid_xyz', 'grid_rpw',
    'roll', 'pitch', 'yaw',
    'k1', 'k2', 'p1', 'p2', 'k3',
    'Z', 'X', 'Y',
    'pinhole_camera_info'
]