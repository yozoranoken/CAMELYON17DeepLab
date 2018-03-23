#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import namedtuple
from random import randint

from matplotlib import pyplot as plt
import numpy as np
import openslide
from PIL import Image
from PIL import ImageDraw
from skimage import img_as_ubyte
from skimage import morphology
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


parser = ArgumentParser(
    description='Extract patches from the WSI tiff images.'
)

parser.add_argument(
    '--data-dir-path',
    help='Path to the directory containing the data folders.',
    required=True,
    type=Path,
    metavar='DATA_DIR_PATH',
)

parser.add_argument(
    '--list-file-path',
    help='Path to the list file containing the filenames of the WSIs.',
    required=True,
    type=Path,
    metavar='LIST_FILE_PATH',
)

parser.add_argument(
    '--tif-dir-name',
    help='Name of the directory containing the WSI tiffs.',
    required=True,
    type=str,
    metavar='TIF_DIR_NAME',
)

parser.add_argument(
    '--label-dir-name',
    help='Name of the directory containing the labels.',
    required=True,
    type=str,
    metavar='LABEL_DIR_NAME',
)

parser.add_argument(
    '--output-path',
    help='Path to write output directory.',
    required=True,
    type=Path,
    metavar='OUTPUT_PATH',
)



DOWNSAMPLING_LEVEL = 5
SMALL_OBJECT_AREA = 512
SMALL_HOLE_AREA = 196
class WSIData:
    GROUP_METASTASES = 'metastases'
    GROUP_NORMAL = 'normal'

    def __init__(self, tif_path, label_path):
        self._tif_path = tif_path
        self._label_path = label_path

        self._roi = None
        self._metastases_mask = None
        self._normal_mask = None
        self._read_polygon()

    def _read_polygon(self):
        tree = ET.parse(str(self._label_path))
        root = tree.getroot()

        normal = []
        metastases = []
        annotations = root[0]
        for annotation in annotations:
            polygon = []
            coordinates = annotation[0]
            for coord in coordinates:
                cx = float(coord.attrib['X'])
                cy = float(coord.attrib['Y'])
                polygon.append((cx, cy))

            if (annotation.attrib['PartOfGroup'] ==
                    self.__class__.GROUP_METASTASES):
                metastases.append(polygon)
            elif (annotation.attrib['PartOfGroup'] ==
                    self.__class__.GROUP_NORMAL):
                normal.append(polygon)

        self._metastases_label = metastases
        self._normal_label = normal
        self._slide = openslide.OpenSlide(str(self._tif_path))

    @property
    def tif_path(self):
        return self._tif_path

    def _downsample_label_coords(self, labels, level):
        ds = self._slide.level_downsamples[level]
        round10s = lambda t: (round(t[0] / ds), round(t[1] / ds))
        return [[round10s(cell) for cell in r]
                for r in labels]

    def get_metastases_label(self, level=0):
        return self._downsample_label_coords(self._metastases_label,
                                             level)

    def get_normal_label(self, level=0):
        return self._downsample_label_coords(self._normal_label,
                                             level)

    @staticmethod
    def _coords2mask(dim, pos, neg=None):
        mask = Image.new('RGB', dim)
        label_draw = ImageDraw.Draw(mask, 'RGB')

        for annotation in pos:
            label_draw.polygon(annotation, fill=(255, 255, 255))

        for annotation in neg:
            label_draw.polygon(annotation, fill=(0, 0, 0))

        del label_draw

        return mask

    def get_metastases_mask(self, level=DOWNSAMPLING_LEVEL):
        if self._metastases_mask is None:
            dim = self._slide.level_dimensions[level]
            pos = self.get_metastases_label(level)
            neg = self.get_normal_label(level)
            self._metastases_mask = np.array(
                self._coords2mask(dim, pos, neg).convert('L'),
                dtype=bool,
            )

        return self._metastases_mask

    def get_normal_mask(self, level=DOWNSAMPLING_LEVEL):
        if self._normal_mask is None:
            roi = self.roi
            metastases = self.get_metastases_mask(level)
            self._normal_mask = np.bitwise_xor(roi, metastases)

        # f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
        # axarr[0].imshow(roi)
        # axarr[1].imshow(normal)
        # plt.show()
        return self._normal_mask

    @property
    def roi(self):
        if self._roi is None:
            slide_img = self._slide.read_region(
                location=(0, 0),
                level=DOWNSAMPLING_LEVEL,
                size=self._slide.level_dimensions[DOWNSAMPLING_LEVEL],
            ).convert('RGB')

            img_rgb = np.array(slide_img, dtype=np.uint8)
            img_hsv = rgb2hsv(img_rgb)
            channel_h = img_hsv[:, :, 0]
            channel_s = img_hsv[:, :, 1]

            thresh_h = threshold_otsu(channel_h)
            thresh_s = threshold_otsu(channel_s)
            binary_h = channel_h > thresh_h
            binary_s = channel_s > thresh_s
            binary = np.bitwise_and(binary_h, binary_s)

            binary = morphology.remove_small_objects(binary, SMALL_OBJECT_AREA)
            binary = morphology.remove_small_holes(binary, SMALL_HOLE_AREA)

            # f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
            # axarr[0].imshow(img_rgb)
            # axarr[1].imshow(binary)

            # plt.imshow(binary)
            # plt.show()

            self._roi = binary
        return self._roi





def get_true_points_2D(mat):
    points = []
    for j, row in enumerate(mat):
        for i, col in enumerate(row):
            if col:
                points.append((i, j))
    return points



def main(args):
    print('[Running patch extraction...]')
    print()

    wsi_data = []

    # Read TIF
    # Read XML Tumor Label
    print('Reading wsi_filenames...')
    with open(str(args.data_dir_path / args.list_file_path)) as list_file:
        tif_dir_path = args.data_dir_path / args.tif_dir_name
        label_dir_path = args.data_dir_path / args.label_dir_name
        for wsi_filename in list_file:
            data = WSIData(
                tif_path=(tif_dir_path / (wsi_filename.strip() + '.tif')),
                label_path=(label_dir_path / (wsi_filename.strip() + '.xml')),
            )
            wsi_data.append(data)

    for data in wsi_data:
        # Get ROI pixels from TIF on low resolution
        # Subtract Tumor Label from Tissue mask
        # Store ROI pixels and tumor pixels
        normal_points = get_true_points_2D(data.get_normal_mask())
        metastases_points = get_true_points_2D(data.get_metastases_mask())

        break







    # 50:50 select a class to extract a patch from
    # Get size coordinates of extracted patch and clip
    #   to tumor label: Patch label generation
    # Save patch and label


if __name__ == '__main__':
    main(parser.parse_args())
