#! /usr/bin/env python3
import os
from argparse import ArgumentParser
from pathlib import Path

from core import parse_dataset
import matplotlib as mpl
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'


parser = ArgumentParser(
    description='Create Thumbnails for visual inspection of WSIs.',
)

parser.add_argument(
    '--data-list-file',
    help='Path to the file containing paths to the WSIs and their Labels',
    type=Path,
    required=True,
    metavar='DATA_LIST_FILE_PATH',
)

parser.add_argument(
    '--output-parent-dir',
    help='Parent directory to write output directory.',
    type=Path,
    default=os.getcwd(),
    metavar='OUTPUT_PARENT_DIR',
)

parser.add_argument(
    '--output-folder-name',
    help='Name of output directory',
    type=str,
    default='WSIThumbs',
    metavar='OUTPUT_FOLDER_NAME',
)

parser.add_argument(
    '--level-downsample',
    help='Level to read from the WSI.',
    type=int,
    default=5,
    metavar='LEVEL_DOWNSAMPLE',
)


def main(args):
    plt.switch_backend('agg')
    output_dir = args.output_parent_dir / args.output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_data = parse_dataset(args.data_list_file)

    level = args.level_downsample

    while wsi_data:
        wd = wsi_data.pop(0)
        print('Writing {}...'.format(wd.name), end='\r')
        general_roi = wd.get_roi_mask(level)
        metastases_mask = wd.get_metastases_mask(level)
        merged_mask = np.full(general_roi.shape, 0)
        merged_mask[np.where(general_roi)] = 1
        merged_mask[np.where(metastases_mask)] = 2
        image = wd.get_full_wsi_image(level)

        fig, axarr = plt.subplots(1, 2, sharex=True, sharey=True)
        plt.suptitle(wd.name, fontsize=20)
        axarr[0].imshow(image)
        axarr[1].imshow(merged_mask, cmap='magma')
        fig.savefig(
            str(output_dir / (wd.name + '.png')),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close(fig)
        wd.close()
        del wd


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
