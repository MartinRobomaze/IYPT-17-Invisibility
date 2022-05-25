import cv2
import os
import h5py
import numpy as np

crop_coords = {
    '10cm': ((1104, 2177), (1104 + 676, 2177 + 287)),
    '15cm': ((1103, 2181), (1103 + 700, 2181 + 290)),
    '20cm': ((1092, 2200), (1092 + 734, 2200 + 288)),
    '30cm': ((1001, 2198), (1001 + 844, 2198 + 282)),
    '40cm': ((940, 2216), (940 + 992, 2216 + 284)),
    '50cm': ((794, 2226), (794 + 1250, 2226 + 284)),
    '60cm': ((622, 2220), (622 + 1626, 2220 + 298)),
}

bg_coords = {
    '10cm': (1441, 2084),
    '15cm': (1438, 2102),
    '20cm': (1459, 2134),
    '30cm': (1405, 2097),
    '40cm': (1422, 2130),
    '50cm': (1416, 2100),
    '60cm': (1363, 2119),
}


def color_diff(color1, color2):
    dR = (color1[0] - color2[0]) / 255
    dG = (color1[1] - color2[1]) / 255
    dB = (color1[2] - color2[2]) / 255

    return np.sqrt(dR**2 + dG**2 + dB**2)


def process_image(im_filename, dist):
    img = cv2.imread(im_filename)

    bg_kernel_size = (11, 11)

    bg_color = [0, 0, 0]

    sum = [0, 0, 0]

    for x in range(bg_coords[dist][0] - bg_kernel_size[0] // 2, bg_coords[dist][0] + bg_kernel_size[0] // 2 + 1):
        for y in range(bg_coords[dist][1] - bg_kernel_size[1] // 2, bg_coords[dist][1] + bg_kernel_size[1] // 2 + 1):
            sum[0] += img[x, y][0]
            sum[1] += img[x, y][1]
            sum[2] += img[x, y][2]

    bg_color[0] = sum[0] / (bg_kernel_size[0] * bg_kernel_size[1])
    bg_color[1] = sum[1] / (bg_kernel_size[0] * bg_kernel_size[1])
    bg_color[2] = sum[2] / (bg_kernel_size[0] * bg_kernel_size[1])

    img = img[crop_coords[dist][0][1]:crop_coords[dist][1][1], crop_coords[dist][0][0]:crop_coords[dist][1][0]]
    bg_threshold = 0.18

    diff = []

    for x in range(len(img)):
        for y in range(len(img[x])):
            diff_xy = color_diff(img[x, y], bg_color)
            if diff_xy > bg_threshold:
                diff.append(diff_xy)

    diff = np.array(diff)

    try:
        idx = np.argpartition(diff, -25)[-25:]
    except ValueError:
        return 0

    return np.average(diff[idx])


data_path = 'data/photos'

box_sizes = np.arange(5, 36, 5)

data_file = h5py.File('data/photos.hdf5', 'w')

for distance in os.listdir(data_path):
    curr_dir = '{}/{}'.format(data_path, distance)

    data_file.create_group(distance)

    for sheet_idx in os.listdir(curr_dir):
        curr_dir = '{}/{}/{}'.format(data_path, distance, sheet_idx)
        data_file[distance].create_group(sheet_idx)

        for orientation in os.listdir(curr_dir):
            curr_dir = '{}/{}/{}/{}'.format(data_path, distance, sheet_idx, orientation)
            data_file[distance][sheet_idx].create_group(orientation)
            data_file[distance][sheet_idx][orientation].create_dataset('blur', len(box_sizes))

            for i, img_filename in enumerate(os.listdir(curr_dir)):
                var = process_image(curr_dir + '/' + img_filename, distance)

                data_file[distance][sheet_idx][orientation]['blur'][i] = var

                print(data_file[distance][sheet_idx][orientation]['blur'][i], img_filename)

data_file.close()
