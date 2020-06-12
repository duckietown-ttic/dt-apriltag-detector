SAMPLE_IDX = 12

import carnivalmirror as cm
import cv2
import dt_apriltags
import math
import numpy as np
import os.path
import pandas as pd

im_path = os.path.normpath(os.path.join(os.path.abspath(__file__), "../samples/sample_%05d.png" % SAMPLE_IDX))
print(im_path)
im = cv2.imread(im_path, cv2.IMREAD_COLOR)
# print(im)
cv2.imshow('input', im)
df = pd.read_csv("datasheet.csv", index_col='idx')

p = df.loc[SAMPLE_IDX]
print(p)
cal = cm.calibration.Calibration(K=[p['fx'], p['fy'], p['cx'], p['cy']],
                                 D=[p['k1'], p['k2'], p['p1'], p['p2'], p['k3']],
                                 width=640, height=480)

im_rect = cal.rectify(im, mode='standard')
cv2.imshow('rect', im_rect)

d = dt_apriltags.Detector(families='tag36h11',
                          nthreads=4,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=0.25,
                          debug=0)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


new_params = [cal.new_camera_matrix[0][0], cal.new_camera_matrix[1][1], cal.new_camera_matrix[0][2],
              cal.new_camera_matrix[1][2]]
tags = d.detect(cv2.cvtColor(im_rect, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True,
                camera_params=new_params, tag_size=0.065)
for t in tags:
    print('tag_id', t.tag_id)
    print('tvec', t.pose_t)
    print('rvec', rotationMatrixToEulerAngles(t.pose_R))

cv2.waitKey(0)  # waits until a key is pressed
