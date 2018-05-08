#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
import numpy as np
from openslide import OpenSlide
from scipy import ndimage as nd
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize


_OUTPUT_DATA_DIRNAME = 'data'
_OUTPUT_LABELS_DIRNAME = 'labels'
_PATCH_SIDE = 768
_PATCH_DIM = _PATCH_SIDE, _PATCH_SIDE
_STRIDE = _PATCH_SIDE // 2
_LEVEL = 1
_SAMPLE_LEVEL = 5
_FP_AREA_THRESHOLD = 0.25

_PATCH_SIDE_0 = _PATCH_SIDE * 2**_LEVEL
_PATCH_DIM_0 = _PATCH_SIDE_0, _PATCH_SIDE_0


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--wsi-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--masks-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--semantic-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-parent-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-folder-name',
        type=str,
        default='PatchesFPBootstrap',
    )

    parser.add_argument(
        '--excludes-list',
        type=Path,
    )

    return parser.parse_args()


def init_label(label, label_img):
    label[np.where(label_img == 1)] = True
    label[np.where(label_img == 2)] = False


def get_label(slide, slide_name, masks_dir):
    h, w = slide.level_dimensions[_SAMPLE_LEVEL]
    label_mask = np.full((w, h), False)
    label_slide = None

    label_path = masks_dir / f'{slide_name}_Mask.tif'

    if label_path.is_file():
        label_slide = OpenSlide(str(label_path))
        label_dim = label_slide.level_dimensions[_SAMPLE_LEVEL]
        label_img = label_slide.read_region((0, 0), _SAMPLE_LEVEL, label_dim)
        label_img = np.array(label_img.convert('L'))
        init_label(label_mask, label_img)

    return label_mask, label_slide


def get_semantic(semantic_path):
    semantic_img = imread(str(semantic_path), as_gray=True) * 255
    distance = nd.distance_transform_edt(255 - semantic_img)
    Threshold = 75/(0.243 * 2**_SAMPLE_LEVEL * 2)
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    return filled_image.astype(bool)


def read_label_patch_0(label_slide, coords):
    label = np.full(_PATCH_DIM, False)
    if label_slide is not None:
        label_img = label_slide.read_region(coords, 0, _PATCH_DIM_0)
        label_img = np.array(label_img.convert('L'))
        label_img = resize(label_img, _PATCH_DIM, preserve_range=True, order=0)
        init_label(label, label_img)
    label = (label * 1).astype(np.uint8)
    return label


def read_patch_and_label(slide, label_slide, coords):
    x, y = coords
    coords_0 = x * 2**_SAMPLE_LEVEL, y * 2**_SAMPLE_LEVEL
    label = read_label_patch_0(label_slide, coords_0)
    patch = slide.read_region(coords_0, 0, _PATCH_DIM_0).convert('RGB')
    patch = resize(np.array(patch), _PATCH_DIM)
    return patch, label


def extract_patches(
        slide_name,
        slide,
        label_slide,
        false_positive_mask,
        output_data_dir,
        output_labels_dir):
    ds_factor = 2**(_SAMPLE_LEVEL - _LEVEL)
    sampling_stride = _STRIDE // ds_factor
    sample_side = _PATCH_SIDE // ds_factor

    h, w = false_positive_mask.shape
    total = h * w // sampling_stride**2
    count = 0
    for y in range(0, h, sampling_stride):
        for x in range(0, w, sampling_stride):
            sample_patch_h = min(sample_side, h - y)
            sample_patch_w = min(sample_side, w - x)

            fp_mask_patch = false_positive_mask[
                y:(y + sample_patch_h),
                x:(x + sample_patch_w),
            ]

            fp_area_cover = np.sum(fp_mask_patch) / fp_mask_patch.size

            count += 1
            print(f'>> {(count / total * 100):.2f}% done', end='\r')
            if fp_area_cover < _FP_AREA_THRESHOLD:
                continue

            patch, label = read_patch_and_label(slide, label_slide, (x, y))

            filename = f'{slide_name}_{count:08d}'
            imsave(str(output_data_dir / f'{filename}.jpeg'),
                   patch, quality=100)
            imsave(str(output_labels_dir / f'{filename}.png'),
                   label)


def get_exludes(exclude_list_path, excludes):
    with open(str(exclude_list_path)) as exclude_file:
        for exclude_name in exclude_file.readlines():
            excludes.append(exclude_name.strip())


def main(args):
    output_folder = args.output_parent_dir / args.output_folder_name
    output_data_dir = output_folder / _OUTPUT_DATA_DIRNAME
    output_labels_dir = output_folder / _OUTPUT_LABELS_DIRNAME
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    semantic_paths = sorted(args.semantic_dir.glob('*.png'))

    excludes = []
    if args.exclude_list is not None:
        get_exludes(args.exclude_list, excludes)

    for semantic_path in semantic_paths:
        slide_name = semantic_path.stem
        if slide_name in excludes:
            print(f'>> Excluding {slide_namme}')
            continue
        else:
            print(f'>> Processing {slide_name}')

        slide_path = tuple(args.wsi_dir.glob(f'**/{slide_name}.tif'))[0]
        slide = OpenSlide(str(slide_path))
        label_mask, label_slide = get_label(
            slide=slide,
            slide_name=slide_name,
            masks_dir=args.masks_dir,
        )
        semantic = get_semantic(semantic_path)
        false_positive_mask = np.bitwise_xor(label_mask, semantic)
        extract_patches(
            slide_name,
            slide,
            label_slide,
            false_positive_mask,
            output_data_dir,
            output_labels_dir,
        )


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(collect_arguments())
