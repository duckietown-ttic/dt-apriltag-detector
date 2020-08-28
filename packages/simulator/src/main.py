#!/usr/bin/env python
import matplotlib

from apriltag_simulator.camera import Camera
from apriltag_simulator.objects import TexturedRectangle3
from apriltag_simulator.lenses import FishEyeLens

from dt_apriltags import Detector

import numpy as np

import os
import time
import yaml
from PIL import Image

import matplotlib.pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
SIM_DATA_DIR = os.path.join(DATA_DIR, 'simulator')

CAMERA_INFO_FILE = os.path.join(DATA_DIR, 'calibration', 'autobot04.yaml')
CAMERA_INFO = yaml.load(open(CAMERA_INFO_FILE, 'rt'), Loader=yaml.SafeLoader)

FX, _, CX, _, FY, CY, _, _, _ = CAMERA_INFO['camera_matrix']['data']
IMG_W = CAMERA_INFO['image_width']
IMG_H = CAMERA_INFO['image_height']

K1, K2, P1, P2, K3 = CAMERA_INFO['distortion_coefficients']['data']

TAG_W_M = 0.08
TAG_H_M = 0.12

# US Letter
TAG_W_M = 0.2159
TAG_H_M = 0.2794


TAG_IMG = os.path.join(SIM_DATA_DIR, 'tags', 'tag0.usletter.png')

camera = Camera('camera1', FX, FY, CX, CY, IMG_W, IMG_H)

lens = FishEyeLens('lens1', K1, K2, K3, P1, P2)

camera.attach_lens(lens)

img = np.full((IMG_W, IMG_H, 3), 125)

OBJ_XYZ = [0, 0, 0.60]
OBJ_RPY = [0, 0, 0]

s = TexturedRectangle3('square1', TAG_IMG, [TAG_W_M, TAG_H_M], xyz=OBJ_XYZ, rpy=OBJ_RPY)

stime = time.time()
for rgb, p in s:
    u, v = camera.capture(p).astype(np.int32).tolist()
    if 0 <= u < IMG_W and 0 <= v < IMG_H:
        img[u, v] = rgb
print('Done in %.1f secs' % (time.time() - stime))

# grey image
grey_img = np.array(Image.fromarray(img.astype('uint8'), 'RGB').convert('LA'))[:, :, 0].transpose((1, 0))

# detect tag
detector = Detector()
detections = detector.detect(grey_img, True, [FX, FY, CX, CY], tag_size=0.168)
print('Detected %d tags' % len(detections))
for detection in detections:
    print('Error[X]: %.2f cm' % float(100 * abs(detection.pose_t.T[0][0] - OBJ_XYZ[0])))
    print('Error[Y]: %.2f cm' % float(100 * abs(detection.pose_t.T[0][1] - OBJ_XYZ[1])))
    print('Error[Z]: %.2f cm' % float(100 * abs(detection.pose_t.T[0][2] - OBJ_XYZ[2])))
    print('Error: %.2f cm' % (100 * np.linalg.norm(detection.pose_t.T - OBJ_XYZ)))

imgplot = plt.imshow(img.transpose((1, 0, 2)))

# draw detections
for detection in detections:
    skirt = 1
    corners = np.zeros_like(detection.corners)
    corners[detection.corners[:, 0] > detection.center[0], 0] = np.ceil(
        detection.corners[detection.corners[:, 0] > detection.center[0], 0]).astype(np.int32) + skirt
    corners[detection.corners[:, 1] > detection.center[1], 1] = np.ceil(
        detection.corners[detection.corners[:, 1] > detection.center[1], 1]).astype(np.int32) + skirt

    corners[detection.corners[:, 0] <= detection.center[0], 0] = np.floor(
        detection.corners[detection.corners[:, 0] <= detection.center[0], 0]).astype(np.int32) - skirt
    corners[detection.corners[:, 1] <= detection.center[1], 1] = np.floor(
        detection.corners[detection.corners[:, 1] <= detection.center[1], 1]).astype(np.int32) - skirt

    plt.plot(
        np.append(corners[:, 0], [corners[0, 0]]),
        np.append(corners[:, 1], [corners[0, 1]]),
        linewidth=4
    )

plt.show()
