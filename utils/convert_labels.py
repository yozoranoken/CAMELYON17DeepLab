#! /usr/bin/env python3.6
from argparse import ArgumentParser
from pathlib import Path
import warnings
import sys

from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import median
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.io import imsave
from skimage.morphology import disk


_LABEL_FILE_GLOB= '*.png'


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--label-dir',
        type=Path,
        required=True,
        metavar='LABEL_DIR',
    )

    parser.add_argument(
        '--output-parent-dir',
        type=Path,
        required=True,
        metavar='OUTPUT_PARENT_DIR',
    )

    parser.add_argument(
        '--output-folder-name',
        type=str,
        metavar='OUTPUT_FOLDER_NAME',
        default='labels_converted',
    )

    return parser.parse_args()


def main():
    args = collect_arguments()

    output_dir = args.output_parent_dir / args.output_folder_name
    output_dir.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count = sum(1 for _ in args.label_dir.glob(_LABEL_FILE_GLOB))

        for i, label_path in enumerate(sorted(args.label_dir.glob(_LABEL_FILE_GLOB))):
            img = imread(label_path, as_gray=True)
            try:
                t = threshold_otsu(img)
                img = img > t
            except ValueError:
                img = img.astype(bool)

            img = (img * 1).astype(np.uint8)

            imsave(output_dir / label_path.name, img)

            print(f'{((i + 1) / count * 100):0.3f}% done',
                  end=('\n' if i == count - 1 else '\r'))

if __name__ == '__main__':
    main()
