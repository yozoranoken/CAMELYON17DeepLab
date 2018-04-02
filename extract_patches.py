#! /usr/bin/env python3
from argparse import ArgumentParser
import csv
from enum import IntEnum
import gc
from pathlib import Path
from random import randint
from uuid import uuid4

import numpy as np
from PIL import Image
from scipy import io as sio

from core import read_wsi_list_file
from core import get_logger
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

parser.add_argument(
    '--output-path',
    help='Path to write output directory.',
    required=True,
    type=Path,
    metavar='OUTPUT_PATH',
)


_METASTASES_PROBABILITY = 60
_SAMPLE_MULTIPLIER = 3

def get_true_points_2D(mat):
    points = []
    for j, row in enumerate(mat):
        for i, col in enumerate(row):
            if col:
                points.append((i, j))
    return points


def color_augment(pil_img, normalizers, centre):
    images = []
    cv_img = np.array(pil_img)
    for n in normalizers:
        if n == centre:
            images.append(pil_img)
        else:
            new_img = n.transform(cv_img)
            images.append(Image.fromarray(new_img))
    return images


def main(args):
    logger = get_logger('Extract-Patches')
    logger.info('Started patch extraction program.')

    sample_dir = Path('./sampled_tiles_from_centers')
    CENTRE_SAMPLES = tuple(map(
        lambda sample: utils.read_image(str(sample_dir / sample)),
        (
            'centre_0_patient_006_node_1.jpeg',
            'centre_1_patient_024_node_2.jpeg',
            'centre_2_patient_056_node_1.jpeg',
            'centre_3_patient_066_node_1.jpeg',
            'centre_4_patient_097_node_3.jpeg',
        ),
    ))

    NORMALIZERS = tuple(stainNorm_Vahadane.Normalizer()
                    for i in range(len(CENTRE_SAMPLES)))
    logger.info('Fitting stain normalizers.')
    for sample, normalizer in zip(CENTRE_SAMPLES, NORMALIZERS):
        normalizer.fit(sample)

    logger.info('Reading WSI data from list file.')
    wsi_data = read_wsi_list_file(args.data_list_file)

    output_parent_dir = args.output_path / args.output_dir_name
    output_jpegs = output_parent_dir / 'data'
    output_annot_png = output_parent_dir / 'annotations_png'
    output_annot_mat = output_parent_dir / 'annotations_mat'
    output_jpegs.mkdir(parents=True, exist_ok=True)
    output_annot_png.mkdir(parents=True, exist_ok=True)
    output_annot_mat.mkdir(parents=True, exist_ok=True)
    logger.info(('Created output directories: \n' +
                 '    Data: %s\n' +
                 '    PNGMasks: %s\n' +
                 '    MATMasks: %s'),
                str(output_jpegs),
                str(output_annot_png),
                str(output_annot_mat))

    while wsi_data:
        data = wsi_data.pop(0)
        ## Get ROI pixels from TIF on low resolution
        ## Subtract Tumor Label from Tissue mask
        # Store ROI pixels and tumor pixels
        logger.info('Extracting patches frorm %s.', data.name)
        logger.info('[WSI %s] - Building ROIs.', data.name)
        normal_points = get_true_points_2D(data.get_normal_mask())
        metastases_points = get_true_points_2D(data.get_metastases_mask())
        logger.info('[WSI %s] - ROIs done.', data.name)

        is_tumor = data.label_path is not None

        roi_area = ((len(metastases_points) + len(normal_points))
                    * 2**data.get_default_downsampling_level())
        patch_area = data.PATCH_DIM[0] * data.PATCH_DIM[1]
        max_count = int(_SAMPLE_MULTIPLIER * (roi_area // patch_area))
        logger.info('[WSI %s] - Extracting %s patches.', data.name, max_count)

        patch_count = 0
        tumor_patch_count = 0
        normal_patch_count = 0

        while patch_count < max_count:
            logger.info('[WSI %s] - Extracting patch [%s / %s].', data.name,
                        patch_count, max_count)
            if is_tumor and randint(0, 100) <= _METASTASES_PROBABILITY:
                logger.info('[WSI %s] - Sampled near metastases.', data.name)
                point_list = metastases_points
                is_tumor_extracted = True
            else:
                logger.info('[WSI %s] - Sampled in normal region.', data.name)
                point_list = normal_points
                is_tumor_extracted = False

            idx = randint(0, len(point_list) - 1)
            logger.info('[WSI %s] - Reading patch.', data.name)
            region, label_img, label_mat = data.read_region(point_list[idx])

            img_var = np.array(region.convert('L')).var()

            if img_var < 350:
                point_list.pop(idx)
                logger.info('[WSI %s] - Dropping. Low color variance.',
                            data.name)
                continue

            if is_tumor_extracted:
                tumor_patch_count += 1
            else:
                normal_patch_count += 1

            logger.info('[WSI %s] - Color augmentation.', data.name)
            aug = color_augment(region, NORMALIZERS, data.centre)
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

            logger.info('[WSI %s] - Writing patches to disk.', data.name)
            uuid_suffix = str(uuid4()).replace('-', '_')
            stem = data.name + '_' + uuid_suffix
            for i, region in enumerate(aug):
                centre_stem = stem + '_c' + str(i)
                patch = centre_stem + '.jpeg'
                annot_png = centre_stem + '.png'
                annot_mat = centre_stem + '.mat'
                region.save(str(output_jpegs / patch), 'JPEG')
                label_img.save(str(output_annot_png / annot_png), 'PNG')
                sio.savemat(str(output_annot_mat / annot_mat), {
                    'data': label_mat,
                })


            del region
            del aug
            del label_img
            del label_mat
            logger.info('[WSI %s] - Done [%s].', data.name, patch_count)
            patch_count += 1

            logger.info('Freed memory: %s', gc.collect())

        logger.info('Done extracting patches from %s.', data.name)
        logger.info('Tumor: %s', tumor_patch_count)
        logger.info('Normal: %s', normal_patch_count)
        logger.info('TOTAL: %s', patch_count)
        del data
        logger.info('Freed memory: %s', gc.collect())


if __name__ == '__main__':
    main(parser.parse_args())
