#! /usr/bin/env python3
from argparse import ArgumentParser
from collections import namedtuple
import csv
from enum import IntEnum
from pathlib import Path
from random import randint
from uuid import uuid4
from xml.etree import ElementTree as ET

from matplotlib import pyplot as plt
import numpy as np
import openslide
from PIL import Image
from PIL import ImageDraw
import pyclipper
from skimage import img_as_ubyte
from skimage import morphology
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

import stain_utils as utils
import stainNorm_Vahadane


parser = ArgumentParser(
    description='Extract patches from the WSI tiff images.'
)

parser.add_argument(
    '--data-list-file',
    help='Path to the file containing paths to the WSIs and their Labels',
    required=True,
    type=Path,
    metavar='DATA_LIST_FILE_PATH',
)

parser.add_argument(
    '--output-dir-name',
    help='Name of the directory to store subdirectories of output.',
    type=str,
    metavar='OUTPUT_DIR_NAME',
    default='Patches',
)

# parser.add_argument(
#     '--data-dir-path',
#     help='Path to the directory containing the data folders.',
#     required=True,
#     type=Path,
#     metavar='DATA_DIR_PATH',
# )

# parser.add_argument(
#     '--list-file-path',
#     help='Path to the list file containing the filenames of the WSIs.',
#     required=True,
#     type=Path,
#     metavar='LIST_FILE_PATH',
# )

# parser.add_argument(
#     '--tif-dir-name',
#     help='Name of the directory containing the WSI tiffs.',
#     required=True,
#     type=str,
#     metavar='TIF_DIR_NAME',
# )

# parser.add_argument(
#     '--label-dir-name',
#     help='Name of the directory containing the labels.',
#     required=True,
#     type=str,
#     metavar='LABEL_DIR_NAME',
# )

parser.add_argument(
    '--output-path',
    help='Path to write output directory.',
    required=True,
    type=Path,
    metavar='OUTPUT_PATH',
)


class LabelClass(IntEnum):
    normal = 0
    metastases = 1


DOWNSAMPLING_LEVEL = 5
SMALL_OBJECT_AREA = 512
SMALL_HOLE_AREA = 196
PATCH_SIDE = 2048
PATCH_DIM = (PATCH_SIDE, PATCH_SIDE)
METASTASES_PROBABILITY = 70
TUMOR_EXTRACTION_COUNT = 100
NORMAL_EXTRACTION_COUNT = 180
TUMOR_LABEL_COLOR = (255, 182, 0)

CENTRE_SAMPLES = (
    utils.read_image('./sampled_tiles_from_centers/centre_0_patient_006_node_1.jpeg'),
    utils.read_image('./sampled_tiles_from_centers/centre_1_patient_024_node_2.jpeg'),
    utils.read_image('./sampled_tiles_from_centers/centre_2_patient_056_node_1.jpeg'),
    utils.read_image('./sampled_tiles_from_centers/centre_3_patient_066_node_1.jpeg'),
    utils.read_image('./sampled_tiles_from_centers/centre_4_patient_097_node_3.jpeg'),
)

class WSIData:
    GROUP_METASTASES = ('metastases', '_0', '_1')
    GROUP_NORMAL = ('normal', '_2')

    def __init__(self, wsi_name, tif_path, label_path):
        self._name = wsi_name
        self._tif_path = tif_path
        self._label_path = label_path or None
        if self._label_path is not None:
            if not self._label_path.is_file():
                raise IOError('File does not exist.')

        self._slide = openslide.OpenSlide(str(self._tif_path))
        self._roi = None
        self._metastases_mask = None
        self._normal_mask = None
        self._read_polygon()

    def _read_polygon(self):
        normal = []
        metastases = []

        if self._label_path is not None:
            tree = ET.parse(str(self._label_path))
            root = tree.getroot()

            annotations = root[0]
            for annotation in annotations:
                polygon = []
                coordinates = annotation[0]
                for coord in coordinates:
                    cx = round(float(coord.attrib['X']))
                    cy = round(float(coord.attrib['Y']))
                    polygon.append((cx, cy))

                if (annotation.attrib['PartOfGroup'] in
                        self.__class__.GROUP_METASTASES):
                    metastases.append(polygon)
                elif (annotation.attrib['PartOfGroup'] in
                        self.__class__.GROUP_NORMAL):
                    normal.append(polygon)

        self._metastases_label = metastases
        self._normal_label = normal

    @property
    def name(self):
        return self._name

    @property
    def tif_path(self):
        return self._tif_path

    @property
    def label_path(self):
        return self._label_path

    def _downsample_label_coords(self, labels, level):
        if not labels:
            return []

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
            label_draw.polygon(annotation, fill='white')

        if neg:
            for annotation in neg:
                label_draw.polygon(annotation, fill=(0, 0, 0))

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
        # axarr[1].imshow(self._normal_mask)
        # plt.show()
        return self._normal_mask

    def _make_coordinates_relative(self, src_pt, regions):
        return [[(x - src_pt[0], y - src_pt[1]) for x, y in region]
                for region in regions]

    def _clip_patch(self, subject, patch_coord, patch_dim):
        w, h = patch_dim
        clipper = (
            (0, 0),
            (w, 0),
            (w, h),
            (0, h),
        )

        subject_relative = self._make_coordinates_relative(
            patch_coord,
            subject,
        )

        pc = pyclipper.Pyclipper()
        pc.AddPath(clipper, pyclipper.PT_CLIP, True)
        pc.AddPaths(subject_relative, pyclipper.PT_SUBJECT, True)

        solution = pc.Execute(
            pyclipper.CT_INTERSECTION,
            pyclipper.PFT_EVENODD,
            pyclipper.PFT_EVENODD,
        )

        solution = [[tuple(c) for c in s] for s in solution]
        return solution


    def read_region(self, ds_coord, ds_level=DOWNSAMPLING_LEVEL):
        ds_factor = round(self._slide.level_downsamples[ds_level])
        w, h = self._slide.dimensions
        pad = PATCH_SIDE // 2
        cx, cy = ds_coord
        cx = (cx * ds_factor) + pad * randint(-1, 1) + randint(0, ds_factor)
        cy = (cy * ds_factor) + pad * randint(-1, 1) + randint(0, ds_factor)
        cx = max(pad, min(cx, w - pad))
        cy = max(pad, min(cy, h - pad))

        region = self._slide.read_region((cx, cy), 0, PATCH_DIM).convert('RGB')

        patch_meta = []
        patch_norm = []
        if self._metastases_label:
            patch_meta = self._clip_patch(
                self._metastases_label,
                (cx, cy),
                PATCH_DIM,
            )

            if self._normal_label:
                patch_norm = self._clip_patch(
                    self._normal_label,
                    (cx, cy),
                    PATCH_DIM,
                )
        label = self._coords2mask(PATCH_DIM, patch_meta, patch_norm)
        return region, label

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

NORMALIZERS = (
    stainNorm_Vahadane.Normalizer(),
    stainNorm_Vahadane.Normalizer(),
    stainNorm_Vahadane.Normalizer(),
    stainNorm_Vahadane.Normalizer(),
    stainNorm_Vahadane.Normalizer(),
)


def color_augment(pil_img):
    images = []
    cv_img = np.array(pil_img)
    for n in NORMALIZERS:
        new_img = n.transform(cv_img)
        images.append(Image.fromarray(new_img))
    return images

def main(args):
    print('> Running patch extraction')

    wsi_data = []

    # Read TIF
    # Read XML Tumor Label
    print('  > Reading wsi_filenames')

    ## Old path reading
    # with open(str(args.data_dir_path / args.list_file_path)) as list_file:
    #     tif_dir_path = args.data_dir_path / args.tif_dir_name
    #     label_dir_path = args.data_dir_path / args.label_dir_name

    #     for wsi_filename in list_file:
    #         data = WSIData(
    #             tif_path=(tif_dir_path / (wsi_filename.strip() + '.tif')),
    #             label_path=(label_dir_path / (wsi_filename.strip() + '.xml')),
    #         )
    #         wsi_data.append(data)

    ## New Path reading
    with open(str(args.data_list_file)) as csvfile:
        csvreader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        for line in csvreader:
            tif_path, label_path = line
            tif_path = Path(tif_path)
            label_path = (label_path or None) and Path(label_path)
            data = WSIData(wsi_name=tif_path.stem,
                           tif_path=tif_path,
                           label_path=label_path)
            wsi_data.append(data)

    print('  > Created output directories')
    output_parent_dir = args.output_path / args.output_dir_name
    output_jpegs = output_parent_dir / 'data'
    output_annot = output_parent_dir / 'labels'
    output_jpegs.mkdir(parents=True, exist_ok=True)
    output_annot.mkdir(parents=True, exist_ok=True)


    for data in wsi_data:
        ## Get ROI pixels from TIF on low resolution
        ## Subtract Tumor Label from Tissue mask
        # Store ROI pixels and tumor pixels
        print('  > Extracting patches from', data.name)
        print('    > Building ROIs')
        normal_points = get_true_points_2D(data.get_normal_mask())
        metastases_points = get_true_points_2D(data.get_metastases_mask())

        is_tumor = data.label_path is not None

        max_count = (TUMOR_EXTRACTION_COUNT if is_tumor
                     else NORMAL_EXTRACTION_COUNT)

        patch_count = 0
        while patch_count < max_count:
            print('    > Extracting patch {}'.format(patch_count))
            if is_tumor and randint(0, 100) <= METASTASES_PROBABILITY:
                point_list = metastases_points
            else:
                point_list = normal_points

            idx = randint(0, len(point_list) - 1)
            region, label = data.read_region(point_list[idx])

            img_var = np.array(region.convert('L')).var()

            if img_var < 350:
                point_list.pop(idx)
                print('      > Dropped patch {}... Restarting'.format(patch_count))
                continue

            print('      > Performing color augmentation')
            aug = color_augment(region)
            # f, axarr = plt.subplots(2, 6, sharex=True, sharey=True)
            # axarr[0, 0].imshow(region)
            # axarr[0, 1].imshow(label)
            # axarr[0, 2].imshow(aug[0])
            # axarr[0, 3].imshow(aug[1])
            # axarr[1, 0].imshow(aug[2])
            # axarr[1, 1].imshow(aug[3])
            # axarr[1, 2].imshow(aug[4])

            # plt.show()
            # break

            print('      > Writing patches to disk')
            uuid_suffix = str(uuid4()).replace('-', '_')
            stem = data.name + '_' + uuid_suffix
            for i, region in enumerate(aug):
                centre_stem = stem + '_c' + str(i)
                patch = centre_stem + '.jpeg'
                annot = centre_stem + '.png'
                region.save(str(output_jpegs / patch), 'JPEG')
                label.save(str(output_annot / annot), 'PNG')
            print('      > Finished patch {}'.format(i))

            patch_count += 1

            # f, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
            # axarr[0].imshow(region)
            # axarr[1].imshow(label)
            # plt.show()

        # 50:50 select a class to extract a patch from
        # label_class = randint(0, 1)

    # Get size coordinates of extracted patch and clip
    #   to tumor label: Patch label generation
    # Save patch and label


if __name__ == '__main__':
    print('> Initializing color normalizers')
    for sample, normalizer in zip(CENTRE_SAMPLES, NORMALIZERS):
        normalizer.fit(sample)

    main(parser.parse_args())
