#! /usr/bin/env python3
from argparse import ArgumentParser
import gc
from math import ceil
import os
from pathlib import Path
from random import randint
from uuid import uuid4
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import measurements
from skimage import transform
from skimage.io import imsave

from core import parse_dataset
from core import get_logger
from core import get_normalizers
from core import image_variance
import stain_utils as utils
import stainNorm_Vahadane


_SAMPLE_SIZES = {
    'normal': {
        'positive': 0,
        'negative': 200,
    },
    'tumor': {
        'positive': 2000,
        'negative': 200,
    }
}
_MIN_TUMOR_PATCHES = 20
_MASK_LEVEL = 5
_FILENAME = '{name}_{uuid}_{idx:05d}_{centre}.{ext}'
_OUTPUT_TUMOR_DIRNAME = 'tumor'
_OUTPUT_NORMAL_DIRNAME = 'normal'
_PATCHES_DIRNAME = 'data'
_LABELS_DIRNAME = 'labels'
_COLOR_VARIANCE_THRESHOLD = 0.05


def collect_arguments():
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
        '--output-parent-dir',
        help='Path to write output directory.',
        required=True,
        type=Path,
        metavar='OUTPUT_PARENT_DIR',
    )

    parser.add_argument(
        '--output-folder-name',
        help='Name of the directory to store subdirectories of output.',
        type=str,
        metavar='OUTPUT_FOLDER_NAME',
        default='Patches',
    )

    parser.add_argument(
        '--patch-side',
        help='Length of the side of the square patch.',
        type=int,
        metavar='PATCH_SIDE',
        default=768,
    )

    parser.add_argument(
        '--normalization-batch-size',
        help='Size of a batch for color normalization.',
        type=int,
        metavar='NORMALIZATION_BATCH_SIZE',
        default=8,
    )

    parser.add_argument(
        '--level',
        help='Level to sample wsis.',
        type=int,
        metavar='LEVEL',
        default=1,
    )

    return parser.parse_args()


def transform_patch_and_label(patch, label, crop_side):
    h, w, _ = patch.shape
    assert h == w, 'Patches must have same height and width.'
    assert crop_side < h, 'Crop side must be smaller than patch shape.'

    # rotation_angle
    # do_transpose
    # do_flip_ud
    # do_flip_lr
    trans = {
        'rotation_angle': np.random.uniform(high=360),
        'do_transpose': np.random.choice([True, False]),
        'do_flipud': np.random.choice([True, False]),
        'do_fliplr': np.random.choice([True, False]),
    }
    patch_t, label_t = patch.copy(), label.copy()

    patch_t = transform.rotate(patch_t, trans['rotation_angle'])
    label_t = transform.rotate(label_t, trans['rotation_angle'])

    if trans['do_transpose']:
        patch_t = np.transpose(patch_t, (1, 0, 2))
        label_t = np.transpose(label_t)

    if trans['do_flipud']:
        patch_t = np.flipud(patch_t)
        label_t = np.flipud(label_t)

    if trans['do_fliplr']:
        patch_t = np.fliplr(patch_t)
        label_t = np.fliplr(label_t)

    offset = (w - crop_side) // 2
    end = offset + crop_side
    patch_t = patch_t[offset:end, offset:end, :]
    label_t = label_t[offset:end, offset:end]

    return patch_t, label_t


# def get_normalizers():
#     sample_dir = Path('../sampled_tiles_from_centers')
#     centre_samples = tuple(map(
#         lambda sample: utils.read_image(str(sample_dir / sample)),
#         (
#             'centre_0_patient_006_node_1.jpeg',
#             'centre_1_patient_024_node_2.jpeg',
#             'centre_2_patient_056_node_1.jpeg',
#             'centre_3_patient_066_node_1.jpeg',
#             'centre_4_patient_097_node_3.jpeg',
#         ),
#     ))

#     normalizers = tuple(stainNorm_Vahadane.Normalizer()
#                     for i in range(len(centre_samples)))
#     for sample, normalizer in zip(centre_samples, normalizers):
#         normalizer.fit(sample)

#     return normalizers


def normalize_patches(patches, normalizers, base_centre):
    patch_side = patches[0].shape[0]
    width = patch_side * len(patches)
    batch_base_image = np.full((patch_side, width, 3), 0, dtype=np.float64)

    for i, x in enumerate(range(0, width, patch_side)):
        batch_base_image[:, x:x + patch_side, :] = patches[i]

    for centre, normalizer in normalizers.items():
        if centre == base_centre:
            normalized = batch_base_image
        else:
            normalized = normalizer.transform(batch_base_image)

        yield list(normalized[:, x:x + patch_side, :]
                   for x in range(0, width, patch_side))


def sample_patches(
        sample_size,
        roi_mask,
        output_dir,
        patch_side,
        level,
        normalizers,
        slide,
        batch_size,
        lg):

    sample_scale_factor = 2**_MASK_LEVEL
    sample_patch_side = ceil(2**0.5 * patch_side)
    sample_patch_dim = sample_patch_side, sample_patch_side
    center_offset = ceil((sample_patch_side * 2**level) / 2)
    data_points = np.argwhere(roi_mask) * sample_scale_factor

    centre_count = len(normalizers)
    lg.info('[%s] - Sampling %s patches.',
            slide.name, sample_size * centre_count)


    patches = []
    labels = []
    sample_count = 0
    while sample_count < sample_size:
        idx = np.random.randint(0, data_points.shape[0])
        cy, cx = data_points[idx]
        cy, cx = (cy + randint(0, sample_scale_factor),
                  cx + randint(0, sample_scale_factor))
        cy, cx = cy - center_offset, cx - center_offset
        patch, label = slide.read_region_and_label(
            (cx, cy), level, sample_patch_dim)

        color_variance = image_variance(patch)
        if color_variance < _COLOR_VARIANCE_THRESHOLD:
            continue

        patch, label = transform_patch_and_label(
            patch, label, patch_side)

        patches.append(patch)
        labels.append(label)

        if (sample_count + 1) % batch_size == 0 or sample_count == sample_size - 1:
            uuid = str(uuid4()).replace('-', '_')
            filename_kwargs = {'name': slide.name, 'uuid': uuid}
            for c, batch in enumerate(normalize_patches(patches,
                                                        normalizers,
                                                        slide.centre)):
                filename_kwargs['centre'] = c
                for idx, p in enumerate(batch):
                    filename_kwargs['idx'] = idx

                    patch_filepath = _FILENAME.format(
                        ext='jpeg', **filename_kwargs)
                    patch_filepath = str(output_dir /
                                         _PATCHES_DIRNAME /
                                         patch_filepath)
                    imsave(patch_filepath, p, quality=100)

                    label_filepath = _FILENAME.format(
                        ext='png', **filename_kwargs)
                    label_filepath = str(output_dir /
                                         _LABELS_DIRNAME /
                                         label_filepath)
                    imsave(label_filepath, labels[idx])

            lg.info('[%s] - (%s / %s) done.',
                    slide.name, (sample_count + 1) * centre_count,
                    sample_size * centre_count)
            del batch

            patches = []
            labels = []

        sample_count += 1

def main():
    args = collect_arguments()

    pid = os.getpid()
    print('Running with PID', pid)

    lg = get_logger('Extract-Patches-Random-{}'.format(pid))

    output_dir = args.output_parent_dir / args.output_folder_name
    output_tumor_dir = output_dir / _OUTPUT_TUMOR_DIRNAME
    output_normal_dir = output_dir / _OUTPUT_NORMAL_DIRNAME
    for od in (output_tumor_dir, output_normal_dir):
        (od / _PATCHES_DIRNAME).mkdir(parents=True, exist_ok=True)
        (od / _LABELS_DIRNAME).mkdir(parents=True, exist_ok=True)

    batch_size = args.normalization_batch_size
    patch_side = args.patch_side
    patch_dim = patch_side, patch_side
    level = args.level

    lg.info('Reading dataset...')
    dataset = parse_dataset(args.data_list_file)
    lg.info('Done reading..')

    lg.info('Initializing normalizers...')
    normalizers = get_normalizers()
    lg.info('Done initializing.')

    while dataset:
        slide = dataset.pop(0)
        lg.info('Sampling from %s.', slide.name)

        positive_roi = slide.get_metastases_mask(_MASK_LEVEL)
        negative_roi = np.bitwise_and(
            slide.get_roi_mask(_MASK_LEVEL),
            np.invert(positive_roi),
        )

        positive_area_0 = np.argwhere(positive_roi).shape[0]
        positive_area_0 = positive_area_0 * (2**_MASK_LEVEL)**2
        patch_area_0 = (patch_side * 2**level)**2
        approx_positive_count = round(positive_area_0 / patch_area_0) * 4

        sampling_kwargs = {
            'patch_side': patch_side,
            'level': level,
            'normalizers': normalizers,
            'slide': slide,
            'batch_size': batch_size,
            'lg': lg,
        }
        if slide.label_xml_path is not None:
            lg.info('[%s] - Sampling tumor patches.', slide.name)
            sample_patches(
                max(_MIN_TUMOR_PATCHES,
                    min(approx_positive_count,
                        _SAMPLE_SIZES['tumor']['positive'])),
                positive_roi,
                output_tumor_dir,
                **sampling_kwargs,
            )
            lg.info('[%s] - Done sampling tumor patches.', slide.name)

            if not slide.is_excluded:
                lg.info('[%s] - Sampling normal patches.', slide.name)
                sample_patches(
                    _SAMPLE_SIZES['tumor']['negative'],
                    negative_roi,
                    output_normal_dir,
                    **sampling_kwargs,
                )
                lg.info('[%s] - Done sampling normal patches.', slide.name)
        else:
            lg.info('[%s] - Sampling normal patches.', slide.name)
            sample_patches(
                _SAMPLE_SIZES['normal']['negative'],
                negative_roi,
                output_normal_dir,
                **sampling_kwargs,
            )
            lg.info('[%s] - Done sampling normal patches.', slide.name)

        lg.info('Done slide %s', slide.name)

        del positive_roi
        del negative_roi
        slide.close()

        lg.info('Freed %s', gc.collect())



if __name__ == '__main__':
    main()
