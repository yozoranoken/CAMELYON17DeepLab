#! /usr/bin/env python3.6
from abc import ABC
from abc import abstractmethod
from argparse import ArgumentParser
from enum import Enum
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


_TRAIN_DIRNAME = 'train'


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--feature-vectors',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--fv-length',
        type=int,
        required=True,
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--np-random-state',
        type=int,
        default=690420,
    )

    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
    )

    parser.add_argument(
        '--train-all',
        action='store_true',
    )


    sub_parsers = parser.add_subparsers(dest='method')


    # Random Forest Parser
    rf_parser = sub_parsers.add_parser(
        EnsembleClassifier.Method.RANDOM_FOREST.value)

    rf_parser.add_argument(
        '--model-filename',
        type=str,
        default='rf_model.pkl'
    )

    rf_parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
    )

    rf_parser.add_argument(
        '--n-estimators',
        type=int,
        default=10,
    )

    rf_parser.add_argument(
        '--verbose',
        type=int,
        default=0,
    )

    rf_parser.add_argument(
        '--max-features',
        type=get_max_features_arg,
        default=_DEFAULT_MAX_FEARURE,
    )

    rf_parser.add_argument(
        '--random-state',
        type=int,
        default=690420,
    )


    return parser.parse_args()


_DATA_NAME_COL = 0
_DATA_FEATURE_COL_START = 1
_DATA_FEATURE_COL_END = 18
_DATA_LABEL_COL = 18


_MAX_FEATURES_VALS = 'auto', 'sqrt', 'log2'
_DEFAULT_MAX_FEARURE = _MAX_FEATURES_VALS[0]

def get_max_features_arg(arg):
    val = arg
    if val is not None:
        try:
            val = float(val)
            if val.is_integer():
                val = int(val)
        except ValueError:
            if val not in _MAX_FEATURES_VALS:
                raise ValueError('max_features not an int, float or in ' +
                                 f'{_MAX_FEATURES_VALS}')

    return val


def split_Xy(data, fv_length):
    X = data[:, :fv_length].astype(np.float64)
    y = data[:, -1].astype(np.float64)
    return X, y


class EnsembleClassifier(ABC):
    class Method(Enum):
        RANDOM_FOREST = 'random_forest'

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def score(self, X, y):
        pass


class RandomForest(EnsembleClassifier):
    def __init__(self, args):
        self._clf = RandomForestClassifier(
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            verbose=args.verbose,
        )

    def fit(self, X, y):
        return self._clf.fit(X, y)

    def save(self, path):
        return joblib.dump(self._clf, path)

    def score(self, X, y):
        return self._clf.score(X, y)


_CLF_MAP = {
    EnsembleClassifier.Method.RANDOM_FOREST: RandomForest,
}

def get_classifier(args):
    return _CLF_MAP[EnsembleClassifier.Method(args.method)](args)


def main(args):
    pd_data = pd.read_csv(str(args.feature_vectors), index_col=0)
    data = pd_data.as_matrix()

    if not args.train_all:
        np.random.seed(args.random_state)
        np.random.shuffle(data)
        X, y = split_Xy(data, args.fv_length)
        split = round(X.shape[0] * args.train_split)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
    else:
        X, y = split_Xy(data, args.fv_length)
        X_train, y_train = X, y

    print(f'Train with {X_train.shape[0]} samples, ' +
          f'each with {X_train.shape[1]} features.')


    classifier = get_classifier(args)

    print('>> Training classifier...', end=' ')
    classifier.fit(X_train, y_train)
    print('Done.')

    if not args.train_all:
        print('>> Predicting data...', end=' ')
        score = classifier.score(X_test, y_test)
        print(f'Done. Test data mean accuracy: {score}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(args.output_dir / args.model_filename)
    classifier.save(output_path)
    print(f'>> Saved model to {output_path}.')


if __name__ == '__main__':
    main(collect_arguments())
