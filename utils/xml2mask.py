#! /usr/bin/env python3
from argparse import ArgumentParser
from collections import namedtuple
import csv
import logging
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import multiresolutionimageinterface as mir
import numpy as np
from PIL import Image

from core import get_logger
from core import parse_dataset


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
    '--output-folder-name',
    help='Name of the directory to store subdirectories of output.',
    type=str,
    metavar='OUTPUT_FOLDER_NAME',
    default='WSIAnnotationMasks',
)

parser.add_argument(
    '--output-parent-dir',
    help='Path to write output directory.',
    required=True,
    type=Path,
    metavar='OUTPUT_PARENT_DIR',
)

_LABEL_MAP = {
    'metastases': 1,
    '_0': 1,
    '_1': 1,
    'normal': 2,
    '_2': 2,
}

def main(args):
    logger = get_logger('XML-to-MASK')

    output_dir_path = args.output_parent_dir / args.output_folder_name
    logger.info('Creating output directory at %s', str(output_dir_path))
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info('Reading WSI data objects.')

    # TODO: replace this call
    wsi_data = parse_dataset(args.data_list_file)

    while wsi_data:
        data = wsi_data.pop(0)
        logger.info('Creating mask for %s', data.name)
        reader = mir.MultiResolutionImageReader()

        if not data.tif_path.is_file():
            logger.warning('TIF File not found. Ignoring %s', data.name)
            continue
        mr_image = reader.open(str(data.tif_path))

        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)

        if data.label_xml_path is None:
            logger.info('No annotation exists. Ignoring %s', data.name)
            continue
        elif not data.label_xml_path.is_file():
            logger.warning('Label File not found. Ignoring %s', data.name)
            continue
        xml_repository.setSource(str(data.label_xml_path))

        xml_repository.load()
        annotation_mask = mir.AnnotationToMask()
        output_path = output_dir_path / (data.name + '_Mask.tif')
        annotation_mask.convert(
            annotation_list,
            str(output_path),
            mr_image.getDimensions(),
            mr_image.getSpacing(),
            _LABEL_MAP,
        )
        logger.info('Mask saved for %s at %s', data.name, str(output_path))

        del data


if __name__ == '__main__':
    main(parser.parse_args())

