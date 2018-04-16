#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
from scipy import ndimage as nd
from skimage import measure

from core import parse_dataset


def collect_arguments():
    parser = ArgumentParser(
        description='Produce feature vectors from the WSI masks.',
    )

    parser.add_argument(
        '--data-list-file',
        type=Path,
        required=True,
        help='Path to the csv file containing WSI file information.',
    )

    parser.add_argument(
        '--output-path',
        type=Path,
        required=True,
        help='Path to write output.',
    )

    parser.add_argument(
        '--output-filename',
        type=Path,
        help='The filename',
        default='features.csv',
    )

    return parser.parse_args()


L0_RESOLUTION = 0.243
LEVEL = 5

def compute_region_mask(slide):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    pixelarray = slide.get_metastases_mask(LEVEL).astype(np.uint8) * 255
    distance = nd.distance_transform_edt(255 - pixelarray)
    Threshold = 75 / (L0_RESOLUTION * 2**LEVEL * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    region_mask = measure.label(filled_image, connectivity=2)
    return region_mask


def main():
    args = collect_arguments()

    wsi_data = parse_dataset(args.data_list_file)[:5]

    while wsi_data:
        slide = wsi_data.pop(0)

        if slide.label_xml_path is None:
            continue

        region_mask = compute_region_mask(slide)
        print(measure.regionprops(region_mask))



if __name__ == '__main__':
    main()
