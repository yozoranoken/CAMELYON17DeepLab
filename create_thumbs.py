#! /usr/bin/env python3
from argparse import ArgumentParser
import logging
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from core import read_wsi_list_file
from core import get_logger


parser = ArgumentParser(
    description='Create Thumbnails for visual inspection of WSIs.',
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
    default='WSIThumbs',
)

parser.add_argument(
    '--output-path',
    help='Path to write output directory.',
    required=True,
    type=Path,
    metavar='OUTPUT_PATH',
)


_NORMAL_COLOR = (24, 16, 94)
_TUMOR_COLOR = (206, 28, 105)

def color_image(img_np, mask, color):
    for i, c in enumerate(color):
        channel = img_np[:,:,i]
        channel[mask] = c
        img_np[:,:,i] = channel


def main(args):
    logger = get_logger('WSI-Thumbs-Generator')
    logger.info('Reading WSI data objects.')
    wsi_data = read_wsi_list_file(args.data_list_file)

    output_path = args.output_path / args.output_dir_name
    logger.info('Creating output directory at %s', str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    while wsi_data:
        data = wsi_data.pop(0)
        logger.info('Writing thumbnail for %s', data.name)
        level = 5
        img = data.get_image(level)


        normal_mask = data.get_normal_mask(level)
        merged_mask = np.zeros(normal_mask.shape + (3,), dtype=np.uint8)

        color_image(merged_mask, normal_mask, _NORMAL_COLOR)
        if data.label_path:
            metastases_mask = data.get_metastases_mask(level)
            color_image(merged_mask, metastases_mask, _TUMOR_COLOR)

        mask_img = Image.fromarray(merged_mask)

        w, h = img.size
        image_and_mask = Image.new('RGB', (w * 2, h))
        image_and_mask.paste(img, (0, 0))
        image_and_mask.paste(mask_img, (w, 0))

        filename = data.name + '.jpg'
        image_and_mask.save(str(output_path / filename), 'JPEG')

        del data


if __name__ == '__main__':
    main(parser.parse_args())
