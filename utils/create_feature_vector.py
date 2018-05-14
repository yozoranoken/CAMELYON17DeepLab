#! /usr/bin/env python3.6
from abc import ABC
from abc import abstractmethod
from argparse import ArgumentParser
import csv
from enum import IntEnum
from functools import partial
from functools import reduce
from multiprocessing import Pool
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from skimage import img_as_float
from skimage.io import imread
from skimage import measure

from core import WSILabels


_HEATMAP_LEVEL = 5
_RESOLUTION = 0.243  # µm

_NAME_LABEL = 'label'


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--softmax-dir',
        type=Path,
        required=True
    )

    parser.add_argument(
        '--semantic-dir',
        type=Path,
        required=True
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--name-list',
        type=Path,
    )

    parser.add_argument(
        '--labels',
        type=Path,
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--cluster-algorithm',
        type=str,
        choices=(
            'connect_2',
            'connect_2_no_dilate',
        ),
        default='connect_2',
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=(
            'method_1',
            'method_2',
            'method_3',
        ),
        default='method_1',
    )

    parser.add_argument(
        '--filename',
        type=str,
        default='features.csv'
    )

    return parser.parse_args()


class FeatureVectorCreator(ABC):
    def __init__(self, cluster_algorithm, include_labels=False):
        self._cluster_algorithm = cluster_algorithm
        self._include_labels = include_labels

        if cluster_algorithm == 'connect_2':
            self.connect_regions = self.connect_2
        if cluster_algorithm == 'connect_2_no_dilate':
            self.connect_regions = self.connect_2_no_dilate

    @abstractmethod
    def create(self, softmax, semantic, name, label=None):
        return

    @property
    @abstractmethod
    def names(self):
        return

    @staticmethod
    def connect_2(mask):
        mask = nd.distance_transform_edt(255 - (mask * 255))

        threshold = 75 / (_RESOLUTION * 2**_HEATMAP_LEVEL * 2)  # 75µm is the
                                                                # equivalent size
                                                                # of 5 tumor cells
        mask = mask < threshold
        mask = nd.morphology.binary_fill_holes(mask)

        mask = measure.label(mask, connectivity=2)

        return mask

    @staticmethod
    def connect_2_no_dilate(mask):
        mask = nd.distance_transform_edt(255 - (mask * 255))

        threshold = 75 / (_RESOLUTION * 2**_HEATMAP_LEVEL * 2)  # 75µm is the
                                                                # equivalent size
                                                                # of 5 tumor cells
        mask = mask < threshold
        mask = nd.morphology.binary_fill_holes(mask)

        mask = measure.label(mask, connectivity=2)

        return mask


class FVMethod001(FeatureVectorCreator):
    _SOFTMAX_THRESHOLDS = (0.5,)
    _LOCAL_PROP_KEYS = (
        'area',
        'extent',
        'eccentricity',
        'major_axis_length',
        'mean_intensity',
        'solidity',
        'perimeter',
    )

    _NAME_PATIENT = 'patient'
    _NAME_MAX_VAL = 'maximum_intensity'
    _NAME_TOTAL_AREA = 'total_area'


    def extract_local_features(self, mask, softmax, feature_vector):
        thresh = self.connect_regions(mask)
        props = measure.regionprops(thresh, softmax)

        if not props:
            feature_vector.extend([0] * len(self._LOCAL_PROP_KEYS))
            return

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
        feature_vector.extend([largest[key] for key in self._LOCAL_PROP_KEYS])

        # Feature: total area of connected regions
        feature_vector.append(sum(prop.area for prop in props))

    def get_threshold_masks(self, softmax, semantic):
        for t in self._SOFTMAX_THRESHOLDS:
            yield softmax > t
        yield semantic.astype(bool)

    @property
    def threshold_suffixes(self):
        names = tuple(map(lambda t: str(int(t * 100)),
                          self._SOFTMAX_THRESHOLDS))
        names += ('semantic',)
        return names

    def create(self, softmax, semantic, name, label=None):
        feature_vector = []
        feature_vector.append(name)
        feature_vector.append(np.amax(softmax))  # Feature: heatmap max value

        # for t in self._SOFTMAX_THRESHOLDS:
        #     self.extract_local_features(softmax > t, softmax, feature_vector)

        for thresh in self.get_threshold_masks(softmax, semantic):
            self.extract_local_features(thresh, softmax, feature_vector)

        # self.extract_local_features(semantic.astype(bool), softmax,
        #                             feature_vector)

        if label is not None and self._include_labels:
            feature_vector.append(label)

        return feature_vector


    def suffixed_local_props(self, suffix):
        keys = map(lambda key: f'{key}_{suffix}', self._LOCAL_PROP_KEYS)
        return tuple(keys) + (f'{self._NAME_TOTAL_AREA}_{suffix}',)


    @property
    def names(self):
        names = [self._NAME_PATIENT, self._NAME_MAX_VAL]

        # for t in self._SOFTMAX_THRESHOLDS:
        #     suffix = round(t * 100)
        #     names.extend(self.suffixed_local_props(suffix))
        #
        # names.extend(self.suffixed_local_props('semantic'))

        for suffix in self.threshold_suffixes:
            names.extend(self.suffixed_local_props(suffix))

        if self._include_labels:
            names.append(_NAME_LABEL)

        return names


class FVMethod002(FVMethod001):
    _SOFTMAX_THRESHOLDS = (0.5, 0.9)

    def get_threshold_masks(self, softmax, semantic):
        for t in self._SOFTMAX_THRESHOLDS:
            yield softmax > t

    @property
    def threshold_suffixes(self):
        names = tuple(map(lambda t: str(int(t * 100)),
                          self._SOFTMAX_THRESHOLDS))
        return names


class FVMethod003(FVMethod001):
    _SOFTMAX_THRESHOLDS = (0.5, 0.9)


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


def generate_names(softmax_dir, name_list_path=None):
    names = []
    if name_list_path is None:
        for softmax_path in sorted(softmax_dir.glob('*.png')):
            names.append(softmax_path.stem)
    else:
        with open(str(name_list_path)) as name_list_file:
            for name in name_list_file.readlines():
                names.append(name.strip())

    return names


def create_feature_vector(fv_creator, softmax_dir, semantic_dir, labels, name):
    print(f'>> Processing {name}')

    filename = f'{name}.png'
    softmax = img_as_float(imread(str(softmax_dir / filename), as_gray=True))
    semantic = imread(str(semantic_dir / filename), as_gray=True)
    label = labels and int(labels[name])

    return fv_creator.create(
        name=name,
        softmax=softmax,
        semantic=semantic,
        label=label
    )


def main(args):
    labels = None
    if args.labels is not None:
        labels = load_labels(args.labels)

    cluster_algorithm = args.cluster_algorithm
    fv_creator = None
    if args.method == 'method_1':
        fv_creator = FVMethod001(
            cluster_algorithm=cluster_algorithm,
            include_labels=labels is not None,
        )
    elif args.method == 'method_2':
        fv_creator = FVMethod002(
            cluster_algorithm=cluster_algorithm,
            include_labels=labels is not None,
        )
    elif args.method == 'method_3':
        fv_creator = FVMethod003(
            cluster_algorithm=cluster_algorithm,
            include_labels=labels is not None,
        )

    names = generate_names(args.softmax_dir, args.name_list)

    f = partial(
        create_feature_vector,
        fv_creator,
        args.softmax_dir,
        args.semantic_dir,
        labels,
    )

    with Pool(args.n_jobs) as pool:
        feature_vectors = pool.map(f, names)

    df = pd.DataFrame(
        data=np.array(feature_vectors),
        columns=fv_creator.names,
    )

    csv_path = str(args.output_dir / args.filename)
    df.to_csv(csv_path, index=False)
    print(f'Saved at {csv_path}')


if __name__ == '__main__':
    main(collect_arguments())
