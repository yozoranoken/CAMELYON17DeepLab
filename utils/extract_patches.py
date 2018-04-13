#! /usr/bin/env python3
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os
from uuid import uuid4
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy import io as sio
from skimage import img_as_ubyte
from skimage.color import rgb2gray

from core import get_logger
from core import parse_dataset
from core import save_label
from core import save_patch
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
    default=513,
)

parser.add_argument(
    '--stride',
    help='Stide to be used when sampling.',
    type=int,
    metavar='STRIDE',
    default=256,
)

parser.add_argument(
    '--level-downsample',
    help='Level to sample wsis.',
    type=int,
    metavar='LEVEL_DOWNSAMPLE',
    default=2,
)

parser.add_argument(
    '--info',
    help='Show estimated values before patching',
    action='store_true',
)


_INFO_SAMPLE_COUNT = 50
_TMP_PATH = Path('/tmp')
_SAMPLE_PATCH_FILENAME_TEMPLATE = 'sample_patch_{:03d}.jpeg'
_SAMPLE_LABEL_FILENAME_TEMPLATE = 'sample_label_{:03d}.mat'

def count_patches(slide, level, stride, patch_side):
    pos_args = level, stride, patch_side
    count = sum(1 for _ in slide.get_roi_patch_positions(*pos_args))
    return count


def show_info(args):
    pid = os.getpid()
    print('Running with PID', pid)

    lg = get_logger('Extract-Patches-{}'.format(pid))
    wsi_data = parse_dataset(args.data_list_file)
    slide = wsi_data[0]

    level = args.level_downsample
    side = args.patch_side
    dim = side, side
    patch_count = 0
    pos_args = level, args.stride, args.patch_side
    for i, pos in enumerate(slide.get_roi_patch_positions(*pos_args)):
        if i < 100:
            continue
        patch, label = slide.read_region_and_label(pos, level, dim)

        patch_filename = _SAMPLE_PATCH_FILENAME_TEMPLATE.format(patch_count)
        label_filename = _SAMPLE_LABEL_FILENAME_TEMPLATE.format(patch_count)
        save_patch(patch, _TMP_PATH / patch_filename)
        save_label(label, _TMP_PATH / label_filename)

        if patch_count == _INFO_SAMPLE_COUNT - 1:
            break
        else:
            patch_count += 1

    avg_patch_size_B = 0
    avg_label_size_B = 0
    for i in range(patch_count):
        patch_filename = _SAMPLE_PATCH_FILENAME_TEMPLATE.format(patch_count)
        label_filename = _SAMPLE_LABEL_FILENAME_TEMPLATE.format(patch_count)
        avg_patch_size_B += os.stat(str(_TMP_PATH / patch_filename)).st_size
        avg_label_size_B += os.stat(str(_TMP_PATH / label_filename)).st_size

    avg_patch_size_B /= patch_count
    avg_label_size_B /= patch_count
    avg_data_B = avg_patch_size_B + avg_label_size_B

    lg.info('Average patch size ------------------------- %s kB',
            int(avg_patch_size_B // 1024))
    lg.info('Average label size ------------------------- %s kB',
            int(avg_label_size_B // 1024))

    total_sz_GB = 0
    total_patch_count = 0
    lg.info('Size of data per WSI:')
    while wsi_data:
        slide = wsi_data.pop(0)
        count = count_patches(slide, *pos_args) * 5
        total_patch_count += count
        sz_GB = count * avg_data_B / 1024**3
        total_sz_GB += sz_GB
        s = ''
        s += ' ' * 2
        s += slide.name
        s += ' ' + ('-' * (40 - len(slide.name))) + ' '
        s += '{} tiles ({:.3f} GB); '.format(count, sz_GB)
        s += '[{}w x {}h]'.format(*slide.get_level_dimension(level))
        lg.info(s)
    lg.info('Total: %s tiles (%s GB)', total_patch_count,
            round(total_sz_GB, 4))


def extract_patches(args):
    # create outputdir
    output_dir = args.output_parent_dir / args.output_folder_name
    output_patches_dir = output_dir / 'data'
    output_labels_dir = output_dir / 'labels'
    output_patches_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # initialize normalizers
    pid = os.getpid()
    print('Running with PID', pid)

    lg = get_logger('Extract-Patches-{}'.format(pid))

    sample_dir = Path('../sampled_tiles_from_centers')
    centre_samples = tuple(map(
        lambda sample: utils.read_image(str(sample_dir / sample)),
        (
            'centre_0_patient_006_node_1.jpeg',
            'centre_1_patient_024_node_2.jpeg',
            'centre_2_patient_056_node_1.jpeg',
            'centre_3_patient_066_node_1.jpeg',
            'centre_4_patient_097_node_3.jpeg',
        ),
    ))

    normalizers = tuple(stainNorm_Vahadane.Normalizer()
                    for i in range(len(centre_samples)))
    lg.info('Fitting stain normalizers...')
    for sample, normalizer in zip(centre_samples, normalizers):
        normalizer.fit(sample)
    lg.info('Done fitting.')

    # fetch data set
    lg.info('Reading data list file...')
    wsi_data = parse_dataset(args.data_list_file)
    lg.info('Done reading.')

    # for each slide
    while wsi_data:
        slide = wsi_data.pop(0)

    #   for each pos in slide
        level = args.level_downsample
        side = args.patch_side
        dim = side, side
        patch_count = 0
        pos_args = level, args.stride, args.patch_side

        lg.info('[WSI %s] - Counting patches...', slide.name)
        centre_count = len(centre_samples)
        patch_count = count_patches(slide, *pos_args) * centre_count

        lg.info('[WSI %s] - Extracting %s patches', slide.name, patch_count)
        for i, pos in enumerate(slide.get_roi_patch_positions(*pos_args)):
            idx = (i + 1) * centre_count
            lg.info('[WSI %s] - Patch (%s / %s)', slide.name, idx,
                    patch_count)
    #       get patch and label
            patch, label = slide.read_region_and_label(pos, level, dim)
    #       ignore when low variance

            gray = np.array(Image.fromarray(patch).convert('L'))
            img_var = gray.var()

            if img_var < 350:
                lg.info('[WSI %s] - Dropping. Low color variance.', slide.name)
                continue

            normalized_patches = []
    #       for each normalizer
            lg.info('[WSI %s] - Normalizing...', slide.name)
            for n in normalizers:
                if n == slide.centre:
                    normalized_patches.append(patch)
                else:
    #               normalize patch
                    normalized_patch = n.transform(patch)
                    normalized_patches.append(normalized_patch)
            lg.info('[WSI %s] - Done normalizing.', slide.name)

            lg.info('[WSI %s] - Writing patches to disk.', slide.name)
            uuid_suffix = str(uuid4()).replace('-', '_')
            stem = slide.name + '_' + uuid_suffix
            for i, normalized_patch in enumerate(normalized_patches):
                centre_stem = stem + '_c' + str(i)
                patch_filename = centre_stem + '.jpeg'
                label_filename = centre_stem + '.mat'

                save_patch(normalized_patch,
                           output_patches_dir / patch_filename)
                save_label(label, output_labels_dir / label_filename)
            lg.info('[WSI %s] - Done writing.', slide.name)


def main(args):
    if args.info:
        show_info(args)
    else:
        extract_patches(args)


if __name__ == '__main__':
    main(parser.parse_args())
