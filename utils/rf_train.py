#! /usr/bin/env python3.6
from argparse import ArgumentParser
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
        required=True
    )

    parser.add_argument(
        '--model-root',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--filename',
        type=str,
        default='rf_model.pkl'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=690420,
    )

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=10,
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
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

    parser.add_argument(
        '--max-features',
        type=str,
        default='auto',
    )

    return parser.parse_args()

_DATA_NAME_COL = 0
_DATA_FEATURE_COL_START = 1
_DATA_FEATURE_COL_END = 18
_DATA_LABEL_COL = 18

def parse_data(fv_path):
    names = []
    features = []
    labels = []

    with open(str(fv_path)) as fv_file:
        fv_reader = csv.reader(fv_file)
        for fv in fv_reader:
            names.append(fv[_DATA_NAME_COL])
            features.append(fv[_DATA_FEATURE_COL_START:_DATA_FEATURE_COL_END])
            labels.append(fv[_DATA_LABEL_COL])

    return (np.array(names), np.array(features, dtype=np.float64),
            np.array(labels, dtype=np.float64))

_MAX_FEATURES_VALS = 'auto', 'sqrt', 'log2'

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


def main(args):
    data = pd.read_csv(str(args.feature_vectors), header=0).as_matrix()
    names = data[:, 0]
    X = data[:, 1:-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)

    X_train, y_train, X_test, y_test = None, None, None, None

    if not args.train_all:
        split = round(X.shape[0] * args.train_split)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
    else:
        X_train, y_train = X, y


    rf_classifier = RandomForestClassifier(
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_features=get_max_features_arg(args.max_features),
        verbose=args.verbose,
    )

    print('>> Training classifier...', end=' ')
    rf_classifier.fit(X_train, y_train)
    print('Done.')

    if not args.train_all:
        print('>> Predicting data...', end=' ')
        score = rf_classifier.score(X_test, y_test)
        print(f'Done. Test data mean accuracy: {score}')

    output_dir = args.model_root / _TRAIN_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / args.filename)
    joblib.dump(rf_classifier, output_path)
    print(f'>> Saved model to {output_path}.')


if __name__ == '__main__':
    main(collect_arguments())
