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
    '10cm': (1418, 1800),
    '15cm': (1430, 1784),
    '20cm': (1437, 1785),
    '30cm': (1402, 1735),
    '40cm': (1400, 1686),
    '50cm': (1395, 1599),
    '60cm': (1389, 1455),
}


def process_image(im_filename, dist):
    img = cv2.imread(im_filename)

    return np.mean(cv2.Canny(img, 50, 200))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img[crop_coords[dist][0][1]:crop_coords[dist][1][1], crop_coords[dist][0][0]:crop_coords[dist][1][0]]

    blur = cv2.Laplacian(img, cv2.CV_64F).var()

    return blur


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
