#! /usr/bin/env python3.6
from argparse import ArgumentParser
import csv
from enum import IntEnum
from functools import reduce
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as nd
from skimage import img_as_float
from skimage.io import imread
from skimage import measure


_HEATMAP_LEVEL = 5
_RESOLUTION = 0.243  # µm
_SOFTMAX_THRESHOLDS = 0.5, 0.9

_CSV_LABEL_UID = 'uid'
_CSV_LABEL_MAX_VAL = 'maximum_intensity'
_CSV_LABEL_TOTAL_AREA = 'total_area'
_CSV_LABEL_LABEL = 'label'
_LOCAL_PROP_KEYS = (
    'area',
    'extent',
    'eccentricity',
    'major_axis_length',
    'mean_intensity',
    'solidity',
    'perimeter',
)


class WSILabels(IntEnum):
    NEGATIVE = 0
    ITC = 1
    MICRO = 2
    MACRO = 4

    @classmethod
    def get_value(cls, label):
        return cls.__members__[label.upper()]


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--softmax-dir',
        type=Path,
        required=True
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--labels',
        type=Path,
    )


    parser.add_argument(
        '--exclude-list',
        type=Path,
    )

    parser.add_argument(
        '--filename',
        type=str,
        default='features.csv'
    )

    return parser.parse_args()


def get_exludes(exclude_list_path, excludes):
    with open(str(exclude_list_path)) as exclude_file:
        for exclude_name in exclude_file.readlines():
            excludes.append(exclude_name.strip())


def load_labels(labels_path):
    labels = {}
    with open(str(labels_path)) as labels_file:
        label_reader = csv.DictReader(labels_file)
        for label_entry in label_reader:
            slide_name = label_entry['patient']
            if slide_name[-4:] == '.zip':
                continue

            labels[slide_name[:-4]] = WSILabels.get_value(label_entry['stage'])

    return labels


def connect_regions(mask):
    mask = nd.distance_transform_edt(255 - (mask * 255))

    threshold = 75 / (_RESOLUTION * 2**_HEATMAP_LEVEL * 2)  # 75µm is the
                                                            # equivalent size
                                                            # of 5 tumor cells
    mask = mask < threshold
    mask = nd.morphology.binary_fill_holes(mask)
    mask = measure.label(mask, connectivity=2)
    return mask


def main(args):
    excludes = []
    if args.exclude_list is not None:
        get_exludes(args.exclude_list, excludes)

    if args.labels is not None:
        labels = load_labels(args.labels)

    feature_vectors = []
    for softmax_path in sorted(args.softmax_dir.glob('*.png')):
        stem = softmax_path.stem
        if stem in excludes:
            print(f'>> Excluding {stem}')
            continue
        else:
            print(f'>> Extracting features from {stem}')

        softmax = img_as_float(imread(str(softmax_path), as_gray=True))

        feature_vector = []
        feature_vector.append(stem)
        feature_vector.append(np.amax(softmax))  # Feature: heatmap max value
        for t in _SOFTMAX_THRESHOLDS:
            thresh = connect_regions(softmax > t)
            props = measure.regionprops(thresh, softmax)

            largest = reduce(lambda x, y: x if x.area > y.area else y,
                             props)

            # Features:
            #   1. area
            #   2. extent
            #   3. eccentricity
            #   4. major_axis_length
            #   5. mean_intensity
            #   6. solidity
            #   7. perimeter
            feature_vector.extend([largest[key] for key in _LOCAL_PROP_KEYS])

            # Feature: total area of connected regions
            feature_vector.append(sum(prop.area for prop in props))

        if args.labels is not None:
            feature_vector.append(int(labels[stem]))

        feature_vectors.append(feature_vector)


    csv_labels = [_CSV_LABEL_UID, _CSV_LABEL_MAX_VAL]
    for t in _SOFTMAX_THRESHOLDS:
        suffix = round(t * 100)
        threshold_keys = map(lambda key: f'{key}_{suffix}',
                             _LOCAL_PROP_KEYS)
        csv_labels.extend(list(threshold_keys))
        csv_labels.append('{_CSV_LABEL_TOTAL_AREA}_{suffix}')

    if args.labels is not None:
        csv_labels.append(_CSV_LABEL_LABEL)


    with open(str(args.output_dir / args.filename), 'w') as outfile:
        fv_writer = csv.writer(outfile)
        fv_writer.writerow(csv_labels)
        for fv in feature_vectors:
            fv_writer.writerow(fv)


if __name__ == '__main__':
    main(collect_arguments())
