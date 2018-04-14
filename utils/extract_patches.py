#! /usr/bin/env python3
from argparse import ArgumentParser
from functools import partial
from math import ceil
from multiprocessing import Pool
from pathlib import Path
import os
from uuid import uuid4
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy import io as sio
import skimage
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.io import imsave
from skimage.util import pad

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
    '--large-patch-span',
    help=('Span of the large patch, for batch color normalization. It will ' +
          'be N=patch_side*span; NxN pixels big.'),
    type=int,
    metavar='LARGE_PATCH_SPAN',
    default=4,
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


def extract_patches_v1(args):
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


def color_fill_with_mask(target_np, mask_np, color):
    for i in range(3):
        target_np[:, :, i][np.where(mask_np)] = color[i]


TUMOR_COLOR = (206, 28, 105)
def make_label_rgb(label_mask):
    label_img = np.full(label_mask.shape + (3,), 0, dtype=np.uint8)
    color_fill_with_mask(label_img, label_mask, TUMOR_COLOR)
    return label_img


def aug_rotate(img_np):
    rotated = []
    for i in range(1, 4):
        rotated.append(skimage.transform.rotate(img_np, 90 * i))
    return rotated


def augment(img_np, label_np):
    augmented_images = []
    augmented_labels = []

    augmented_images.extend(aug_rotate(img_np))
    augmented_labels.extend(aug_rotate(label_np))

    img_t_np = np.transpose(img_np, (1, 0, 2))
    label_t_np = np.transpose(label_np, (1, 0, 2))
    augmented_images.append(img_t_np)
    augmented_labels.append(label_t_np)

    augmented_images.extend(aug_rotate(img_t_np))
    augmented_labels.extend(aug_rotate(label_t_np))

    return augmented_images, augmented_labels


_ROI_PATCH_COVER_PERCENTAGE = 0.3
def extract_patches_v2(args):
    # create outputdir
    output_dir = args.output_parent_dir / args.output_folder_name
    output_patches_dir = output_dir / 'data'
    output_labels_dir = output_dir / 'labels'
    output_patches_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # initialize normalizers
    # pid = os.getpid()
    # print('Running with PID', pid)

    # lg = get_logger('Extract-Patches-{}'.format(pid))
    lg = get_logger('Extract-Patches')

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

        lg.info('Sampling from %s', slide.name)

        total_patch_count = 0
        level = args.level_downsample
        ds_level = 5
        side = args.patch_side
        stride = args.stride
        dim = side, side
        wsi_width, wsi_height = slide.get_level_dimension(level)
        large_stride = (args.large_patch_span * side)
        large_patch_side = large_stride - stride + side
        large_patch_dim = large_patch_side, large_patch_side
        upscale_factor = 2**level
        downscale_factor = 2**(ds_level - level)
        roi_patch_cover_count = (_ROI_PATCH_COVER_PERCENTAGE *
                                 (side // downscale_factor)**2)

        roi = slide.get_roi_mask(ds_level)
        metastases_roi = slide.get_metastases_mask(ds_level)

        lg.info('[WSI %s] - W: %s, H: %s', slide.name,
                ceil(wsi_width / large_stride),
                ceil(wsi_height / large_stride))
        lg.info('[WSI %s] - large patch side: %s', slide.name, large_patch_side)


        # x_lvl, y_lvl -- coords relative to specified level
        for j_lvl, y_lvl in enumerate(range(0, wsi_height, large_stride)):
            for i_lvl, x_lvl in enumerate(range(0, wsi_width, large_stride)):
                w_large = min(large_patch_side, wsi_width - x_lvl)
                h_large = min(large_patch_side, wsi_height - y_lvl)
                # w_pad = large_patch_side - w_large
                # h_pad = large_patch_side - h_large
                w_large = (w_large // stride) * stride
                h_large = (h_large // stride) * stride

                if w_large * h_large == 0:
                    # lg.info('[WSI %s] - Insufficient area for a single ' +
                    #         'patch (W:%s, H:%s). Skipping...', slide.name,
                    #         w_large, h_large)
                    continue

                large_dim = (w_large, h_large)

                # read region with lvl_0 coordinates
                x_0 = x_lvl * upscale_factor
                y_0 = y_lvl * upscale_factor
                large_patch, large_label = slide.read_region_and_label(
                    (x_0, y_0),
                    level,
                    large_dim,
                )

                # large_region = pad(
                #     large_region,
                #     pad_width=((0, h_pad), (0, w_pad), (0, 0)),
                #     mode='constant',
                # )

                # large_label = pad(
                #     large_label,
                #     pad_width=((0, h_pad), (0, w_pad)),
                #     mode='constant',
                # )

                x_lvl_ds = x_lvl // downscale_factor
                y_lvl_ds = y_lvl // downscale_factor
                h_large_ds = h_large // downscale_factor
                w_large_ds = w_large // downscale_factor
                large_roi = roi[
                    y_lvl_ds:(y_lvl_ds + h_large_ds),
                    x_lvl_ds:(x_lvl_ds + w_large_ds)
                ]
                large_metastases_roi = metastases_roi[
                    y_lvl_ds:(y_lvl_ds + h_large_ds),
                    x_lvl_ds:(x_lvl_ds + w_large_ds)
                ]

                # print(w_large, h_large)

                roi_cover = np.sum(large_roi)

                if roi_cover < roi_patch_cover_count:
                    # lg.info('[WSI %s] - Insufficient ROI area w/in large ' +
                    #         'patch (%s). Skipping...', slide.name, roi_cover)
                    continue

                # if np.sum(large_metastases_roi) == 0:
                #     continue

                # imgs = (large_patch, large_roi, large_metastases_roi)
                # fig, axes = plt.subplots(nrows=1, ncols=3)
                # for i in range(len(imgs)):
                #     axes[i].imshow(imgs[i])
                # plt.show()
                #
                # continue
                lg.info('[WSI %s] - Sampling from large region (%s, %s)',
                        slide.name, i_lvl, j_lvl)

                normalized_large_patches = []
                lg.info('[WSI %s] - Normalizing...', slide.name)
                for i, n in enumerate(normalizers):
                    if i == slide.centre:
                        normalized_large_patches.append(large_patch)
                    else:
                        normalized_large_patch = n.transform(large_patch)
                        normalized_large_patches.append(normalized_large_patch)
                lg.info('[WSI %s] - Done normalizing.', slide.name)

                # fig, axes = plt.subplots(nrows=3, ncols=2,
                #                          sharex=True, sharey=True)
                # axes[0][0].imshow(normalized_large_patches[0])
                # axes[0][1].imshow(normalized_large_patches[1])
                # axes[1][0].imshow(normalized_large_patches[2])
                # axes[1][1].imshow(normalized_large_patches[3])
                # axes[2][0].imshow(normalized_large_patches[4])
                # plt.show()

                side_ds = side // downscale_factor
                stride_ds = stride // downscale_factor
                # keeps strided patching within boundary
                w_large -= side * (w_large < large_patch_side)
                h_large -= side * (h_large < large_patch_side)

                # fig, axes = plt.subplots(nrows=ceil(h_large / stride),
                #                          ncols=ceil(w_large / stride))
                lg.info('[WSI %s] - Extracting patches from region...', slide.name)
                for j, cy in enumerate(range(0, h_large, stride)):
                    for i, cx in enumerate(range(0, w_large, stride)):
                        cy_ds = cy // downscale_factor
                        cx_ds = cx // downscale_factor
                        patch_roi = large_roi[
                            cy_ds:(cy_ds + side_ds),
                            cx_ds:(cx_ds + side_ds)
                        ]

                        total_px = np.sum(patch_roi)
                        if total_px < roi_patch_cover_count:
                            continue

                        patch_metastases_roi = large_metastases_roi[
                            cy_ds:(cy_ds + side_ds),
                            cx_ds:(cx_ds + side_ds)
                        ]

                        total_px_meta = np.sum(patch_metastases_roi)
                        tumor_cover = (total_px_meta / patch_metastases_roi.size)

                        # print(tumor_cover, total_px_meta, patch_metastases_roi.size)

                        label_mask = large_label[
                            cy:(cy + side),
                            cx:(cx + side),
                        ]
                        label_img = make_label_rgb(label_mask)

                        for centre_id, normalized_large_patch in enumerate(normalized_large_patches):
                            norm_patches = []
                            norm_labels = []

                            norm_patch = normalized_large_patch[
                                cy:(cy + side),
                                cx:(cx + side),
                            ]
                            norm_label = np.copy(label_img)
                            total_patch_count += 1

                            norm_patches.append(norm_patch)
                            norm_labels.append(norm_label)

                            if tumor_cover > 0.0625:
                                aug_images, aug_labels = augment(norm_patch, norm_label)
                                norm_patches.extend(aug_images)
                                norm_labels.extend(aug_labels)
                                total_patch_count += 7

                                # fig, axes = plt.subplots(nrows=2, ncols=len(aug_images))
                                # for i in range(len(images)):
                                #     axes[0][i].imshow(aug_images[i])
                                #     axes[1][i].imshow(aug_labels[i])
                                # plt.show()

                            for patch_id in range(len(norm_patches)):
                                filename = '{name}_lvl{level}_X{X:07d}_Y{Y:07d}_x{x:03d}_y{y:03d}_c{centre}_{patch_id}.png'.format(
                                    name=slide.name,
                                    level=level,
                                    X=x_lvl,
                                    Y=y_lvl,
                                    x=i,
                                    y=j,
                                    centre=centre_id,
                                    patch_id=patch_id,
                                )

                                imsave(str(output_patches_dir / filename),
                                       norm_patches[patch_id])
                                imsave(str(output_labels_dir / filename),
                                       norm_labels[patch_id])
                lg.info('[WSI %s] - Done extracting. ', slide.name)

                        # print(i, j, patch.shape)
                        # a = axes[j][i]
                        # a.get_xaxis().set_ticklabels([])
                        # a.get_yaxis().set_ticklabels([])
                        # a.imshow(patch)
                        # a.imshow(label_img)
                # plt.show()
        lg.info('[WSI %s] - Done sampling from WSI. %s patches sampled.',
                slide.name, total_patch_count)







def main(args):
    if args.info:
        show_info(args)
    else:
        # extract_patches_v1(args)
        extract_patches_v2(args)


if __name__ == '__main__':
    main(parser.parse_args())
