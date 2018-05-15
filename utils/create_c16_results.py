#! /usr/bin/env python3
from argparse import ArgumentParser
from enum import Enum
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from skimage import measure
from skimage import img_as_float
from skimage.io import imread


_LEVEL = 1
_SEMATIC_FILENAME = '*.png'

class IntensityMethod(Enum):
    MEAN = 'mean'
    MAX = 'max'


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--semantic-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--softmax-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=2,
    )

    parser.add_argument(
        '--intensity-method',
        choices=tuple(m for m in IntensityMethod),
        type=IntensityMethod,
        default=IntensityMethod.MEAN,
    )

    parser.add_argument(
        '--folder-name',
        type=str,
        default='c16_results'
    )

    parser.add_argument(
        '--use-edt',
        action='store_true',
    )

    return parser.parse_args()


def preprocess_semantic(semantic_mask, use_edt, resolution=0.243, level=5):
    mask = semantic_mask
    if use_edt:
        mask = nd.distance_transform_edt(1 - (semantic_mask * 1))

        threshold = 75 / (resolution * 2**level * 2)  # 75µm is the
                                                    # equivalent size
                                                    # of 5 tumor cells
        mask = mask < threshold

    mask = nd.morphology.binary_fill_holes(mask)
    mask = measure.label(mask, connectivity=2)
    return mask

def make_results(semantic_dir, softmax_dir, intensity_method, use_edt, name):
    print(f'>> Processing {name}...')

    semantic = imread(semantic_dir / f'{name}.png').astype(bool)
    softmax = img_as_float(imread(softmax_dir / f'{name}.png'))

    labelled_semantic = preprocess_semantic(semantic, use_edt)
    region_props = measure.regionprops(labelled_semantic, softmax)

    results = []

    factor = 2**_LEVEL
    for prop in region_props:
        x, y = prop.centroid
        x, y = x * factor, y * factor

        if intensity_method == IntensityMethod.MEAN:
            p = prop.mean_intensity
        elif intensity_method == IntensityMethod.MAX:
            p = prop.max_intensity

        results.append((p, x, y))

    return pd.DataFrame(np.array(results, dtype='f8,u4,u4'))


def main(args):
    output_dir = args.output_dir / args.folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [pth.stem for pth in
             sorted(args.semantic_dir.glob(_SEMATIC_FILENAME))]

    f = partial(
        make_results,
        args.semantic_dir,
        args.softmax_dir,
        args.intensity_method,
        args.use_edt,
    )

    with Pool(args.n_jobs) as pool:
        all_results = pool.map(f, names)

    for name, results in zip(names, all_results):
        results.to_csv(str(output_dir / f'{name}.csv'), header=False,
                       index=False)


if __name__ == '__main__':
    main(collect_arguments())
