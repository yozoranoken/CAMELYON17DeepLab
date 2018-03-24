#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import openslide


parser = ArgumentParser(
    description='Extract a single patch from a WSI tiff image.'
)

parser.add_argument(
    '--input-path', '-i',
    help='Path to the WSI tiff image.',
    required=True,
    type=Path,
    metavar='INPUT_PATH',
)

parser.add_argument(
    '--output-path', '-o',
    help='Path to output the extracted region.',
    required=True,
    type=Path,
    metavar='OUTPUT_PATH',
)

parser.add_argument(
    '--coord-x', '-X',
    help='X coordinate.',
    required=True,
    type=int,
    metavar='X_COORD',
)

parser.add_argument(
    '--coord-y', '-Y',
    help='Y coordinate.',
    required=True,
    type=int,
    metavar='Y_COORD',
)

parser.add_argument(
    '--side-length', '-S',
    help='Length of patch side.',
    required=True,
    type=int,
    metavar='SIDE_LENGTH',
)


def main(args):
    wsi_slide = openslide.OpenSlide(str(args.input_path))
    coords = args.coord_x, args.coord_y
    dim = args.side_length, args.side_length
    region_img = wsi_slide.read_region(coords, 0, dim).convert('RGB')
    name = args.input_path.stem + str(uuid4()).replace('-', '_') + '.jpeg'
    region_img.save(str(args.output_path / name))


if __name__ == '__main__':
    main(parser.parse_args())
