import carnivalmirror as cm
import cv2
import imgaug.augmenters as iaa
import numpy as np
import os
import os.path
import pandas as pd
import numpy

PATH_TO_TAG_IMAGES = os.path.join(os.path.abspath(__file__), '../apriltag-imgs/tag36h11')
PATH_TO_BG_IMAGES = os.path.normpath(os.path.join(os.path.abspath(__file__), '../tag_bgs'))
IMAGE_RESOLUTION = (480, 640)
ROTATION_RANGES = ([-0.1 * np.pi, -0.1 * np.pi, -0.1 * np.pi], [0.1 * np.pi, 0.1 * np.pi, 0.1 * np.pi])
TRANSLATION_RANGES = ([-0.2, -0.2, 0.05], [0.2, 0.2, 0.5])

ranges = {'fx': (320.0, 480.0),
          'fy': (320.0, 480.0),
          'cx': (320 - 40, 320 + 40),
          'cy': (240 - 20, 240 + 20),
          'k1': (-2.0, 0.0),
          'k2': (0.0, 0.1),
          'p1': (0.000, 0.001),
          'p2': (-0.001, 0.000),
          'k3': (0.0, 0.0)}

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 3)),
    # iaa.SaltAndPepper(p=(0.0, 0.2)),
    iaa.GaussianBlur(sigma=(0, 2)),
    iaa.GammaContrast(gamma=(0, 2)),
    iaa.ChangeColorTemperature((1100, 10000))
    # iaa.MultiplyHueAndSaturation(mul=0.5)
])

####################################################################

output_data = {
    'idx': list(),
    'tag_id': list(),
    'tvec': list(),
    'rvec': list(),
    'fx': list(),
    'fy': list(),
    'cx': list(),
    'cy': list(),
    'k1': list(),
    'k2': list(),
    'p1': list(),
    'p2': list(),
    'k3': list(),
    'p11': list(),
    'p12': list(),
    'p13': list(),
    'p21': list(),
    'p22': list(),
    'p23': list(),
    'p31': list(),
    'p32': list(),
    'p33': list()}


def draw_quad(image, cps, color=(0, 255, 0)):
    cv2.line(image, tuple(cps['top_left'].astype(int)), tuple(cps['top_right'].astype(int)), color, thickness=2)
    cv2.line(image, tuple(cps['top_right'].astype(int)), tuple(cps['bot_right'].astype(int)), color, thickness=2)
    cv2.line(image, tuple(cps['bot_right'].astype(int)), tuple(cps['bot_left'].astype(int)), color, thickness=2)
    cv2.line(image, tuple(cps['bot_left'].astype(int)), tuple(cps['top_left'].astype(int)), color, thickness=2)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


for im_idx in range(1):
    found_suitable_image = False

    while not found_suitable_image:
        files = os.listdir(PATH_TO_BG_IMAGES)
        bg_file = os.path.join(PATH_TO_BG_IMAGES, np.random.choice(files))
        im = cv2.imread(bg_file, cv2.IMREAD_COLOR)

        sampler = cm.sampling.ParameterSampler(ranges, cal_width=IMAGE_RESOLUTION[1], cal_height=IMAGE_RESOLUTION[0])

        calibration = sampler.next()

        tag_id = np.random.randint(0, 587)
        # tag_id = 1
        tag_image_name = 'tag36_11_%05d.png' % tag_id
        tag_image = cv2.imread(os.path.normpath(os.path.join(PATH_TO_TAG_IMAGES, tag_image_name)), cv2.IMREAD_COLOR)

        tvec = np.random.uniform(*TRANSLATION_RANGES)
        rvec = np.random.uniform(*ROTATION_RANGES)
        points = np.array([[-1, 1, 0],
                           [1, 1, 0],
                           [1, -1, 0],
                           [-1, -1, 0]]) * 0.5 * 0.065

        distCoeffs = calibration.get_D()
        cameraMatrix = calibration.get_K(height=IMAGE_RESOLUTION[0])

        projected, _ = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
        proj_pt = lambda pt: \
            cv2.projectPoints(np.array([[pt[0], pt[1], 0.0]]), rvec, tvec, cameraMatrix, distCoeffs)[0][0][
                0]
        # check that all corners are in the image bounds and that the image is not smaller than 5% of the dimensions
        x_min = np.min([projected[i][0][0] for i in range(4)])
        x_max = np.max([projected[i][0][0] for i in range(4)])
        y_min = np.min([projected[i][0][1] for i in range(4)])
        y_max = np.max([projected[i][0][1] for i in range(4)])

        if x_max - x_min < 0.1 * IMAGE_RESOLUTION[1]:
            continue
        if y_max - y_min < 0.1 * IMAGE_RESOLUTION[0]:
            continue
        if x_min < 0 or x_max > IMAGE_RESOLUTION[1] or y_min < 0 or y_max > IMAGE_RESOLUTION[0]:
            continue

        found_suitable_image = True

    cps = {'bot_left': points[0][:2],
           'bot_right': points[1][:2],
           'top_right': points[2][:2],
           'top_left': points[3][:2]}

    # im = np.zeros((IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 3))

    draw_quad(im, cps)

    for i in range(10):
        for j in range(10):
            top_l = cps['top_left'] + (i * 0.1) * (cps['top_right'] - cps['top_left'])
            top_r = cps['top_left'] + ((i + 1) * 0.1) * (cps['top_right'] - cps['top_left'])
            bot_l = cps['bot_left'] + (i * 0.1) * (cps['bot_right'] - cps['bot_left'])
            bot_r = cps['bot_left'] + ((i + 1) * 0.1) * (cps['bot_right'] - cps['bot_left'])
            left_t = cps['top_left'] - (j * 0.1) * (cps['top_left'] - cps['bot_left'])
            left_b = cps['top_left'] - ((j + 1) * 0.1) * (cps['top_left'] - cps['bot_left'])
            right_t = cps['top_right'] - (j * 0.1) * (cps['top_right'] - cps['bot_right'])
            right_b = cps['top_right'] - ((j + 1) * 0.1) * (cps['top_right'] - cps['bot_right'])

            # get corner points for the small square
            inner_cps = {'bot_left': proj_pt(line_intersection((left_b, right_b), (top_l, bot_l))).astype(int),
                         'bot_right': proj_pt(line_intersection((left_b, right_b), (top_r, bot_r))).astype(int),
                         'top_left': proj_pt(line_intersection((left_t, right_t), (top_l, bot_l))).astype(int),
                         'top_right': proj_pt(line_intersection((left_t, right_t), (top_r, bot_r))).astype(int)}

            color = tuple([int(c) for c in tag_image[j, i]])

            cv2.fillPoly(im,
                         np.array([[inner_cps['bot_left'],
                                    inner_cps['bot_right'],
                                    inner_cps['top_right'],
                                    inner_cps['top_left']]]),
                         color)

            # if i==2 and j==5:
            #     cv2.line(im, tuple(top_l.astype(int)), tuple(top_r.astype(int)), (0, 0, 255), thickness=2)
            #     cv2.line(im, tuple(bot_l.astype(int)), tuple(bot_r.astype(int)), (0, 0, 255), thickness=2)
            #     cv2.line(im, tuple(left_b.astype(int)), tuple(left_t.astype(int)), (0, 0, 255), thickness=2)
            #     cv2.line(im, tuple(right_b.astype(int)), tuple(right_t.astype(int)), (0, 0, 255), thickness=2)
            #
            #     draw_quad(im, inner_cps, (255, 0, 0))
    #im_aug = seq(image=im)
    im_aug = im
    if not os.path.exists('raw'):
        os.makedirs('raw')
    cv2.imwrite("raw/%05d_raw.jpg" % im_idx, im_aug, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    output_data['fx'].append(cameraMatrix[0][0])
    output_data['fy'].append(cameraMatrix[1][1])
    output_data['cx'].append(cameraMatrix[0][2])
    output_data['cy'].append(cameraMatrix[1][2])
    output_data['k1'].append(distCoeffs[0])
    output_data['k2'].append(distCoeffs[1])
    output_data['p1'].append(distCoeffs[2])
    output_data['p2'].append(distCoeffs[3])
    output_data['k3'].append(distCoeffs[4])
    output_data['idx'].append(im_idx)
    output_data['tag_id'].append(tag_id)
    output_data['tvec'].append(tvec)
    output_data['rvec'].append(rvec)
    R = cv2.Rodrigues(rvec)
    R = (R[0])
    R = np.identity(3)
    f = np.c_[ R, tvec ]  
    P = np.dot(cameraMatrix , R)
    print(P)
    output_data['p11'].append(P[0][0])
    output_data['p12'].append(P[0][1])
    output_data['p13'].append(P[0][2])
    output_data['p21'].append(P[1][0])
    output_data['p22'].append(P[1][1])
    output_data['p23'].append(P[1][2])
    output_data['p31'].append(P[2][0])
    output_data['p32'].append(P[2][1])
    output_data['p33'].append(P[2][2])
    P = P[:,:3]
    ds = np.identity(3)
    mapx = numpy.ndarray(shape=(IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 1), dtype='float32')
    mapy = numpy.ndarray(shape=(IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 1), dtype='float32')

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R , cameraMatrix , (IMAGE_RESOLUTION[1], IMAGE_RESOLUTION[0]), cv2.CV_32FC1, mapx, mapy)

    res = cv2.remap(im_aug, mapx, mapy, cv2.INTER_NEAREST)

    im_aug = res
    if not os.path.exists('rect'):
        os.makedirs('rect')
    cv2.imwrite("rect/%05d_rect.jpg" % im_idx, im_aug, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

df = pd.DataFrame(output_data)
df.to_csv('datasheet.csv', index=False)
