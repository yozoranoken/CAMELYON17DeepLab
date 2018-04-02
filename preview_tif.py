#! /usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import openslide


parser = ArgumentParser(
    description='Show preview of TIF in pyplot.',
)

parser.add_argument(
    '--path-to-tif',
    help='Path to the TIF file to preview.',
    required=True,
    type=Path,
    metavar='TIF_PATH',
)

parser.add_argument(
    '--level',
    help='Level downsample to view TIF.',
    type=int,
    metavar='LEVEL_DOWNSAMPLE',
    default=5,
)


def main(args):
    slide = openslide.OpenSlide(str(args.path_to_tif))
    img = slide.read_region(
        location=(0, 0),
        level=args.level,
        size=slide.level_dimensions[args.level],
    ).convert('L')

    img_np = np.array(img)
    print(np.unique(img_np))
    # img_np[np.where(img_np == 1)] = 255
    # img_np[np.where(img_np == 2)] = 128
    # print(np.unique(img_np))

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main(parser.parse_args())
