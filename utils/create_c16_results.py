#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
print('Hello')
from scipy import ndimage as nd
from skimage import measure
from skimage.io import imread


_SEMATIC_FILENAME = '*.png'


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
        '--folder-name',
        type=str,
        default='results'
    )

    return parser.parse_args()


def preprocess_semantic_mask(semantic_mask, resolution=0.243, level=5):
    mask = nd.distance_transform_edt(1 - (semantic_mask * 1))

    threshold = 75 / (resolution * 2**level * 2)  # 75µm is the
                                                  # equivalent size
                                                  # of 5 tumor cells
    mask = mask < threshold
    mask = nd.morphology.binary_fill_holes(mask)
    mask = measure.label(mask, connectivity=2)
    return mask


def main(args):
    output_dir = args.output_dir / args.folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for semantic_path in sorted(args.semantic_dir.glob(_SEMATIC_FILENAME)):
        stem = semantic_path.stem
        print(f'>> Processing {stem}...', end=' ')

        softmax_path = args.softmax_dir / f'{stem}.png'

        semantic_img = imread(semantic_path)
        plt.imshow(semantic_img, cmap='inferno')
        plt.show()


        print('Done.')


if __name__ == '__main__':
    main(collect_arguments())
