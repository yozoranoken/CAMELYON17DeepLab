#! /usr/bin/env python3.6
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ensembles import Classifier
from ensembles import RandomForest
from ensembles import get_classifier
from ensembles import write_csv_predictions_vs_ground_truth



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

    parser.add_argument(
        '--model-filename',
        type=str,
    )


    sub_parsers = parser.add_subparsers(dest='method')


    # Random Forest Parser
    rf_parser = sub_parsers.add_parser(
        Classifier.Method.RANDOM_FOREST.value)

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
        type=RandomForest.get_max_features_arg,
        default=RandomForest.DEFAULT_MAX_FEATURE,
    )

    rf_parser.add_argument(
        '--random-state',
        type=int,
        default=690420,
    )

    # Random Forest Parser
    gc_parser = sub_parsers.add_parser(
        Classifier.Method.GC_FOREST.value)

    gc_parser.add_argument(
        '--config',
        type=Path,
        required=True,
    )

    return parser.parse_args()


def split_Xy(data, fv_length):
    X = data[:, :fv_length].astype(np.float64)
    y = data[:, -1].astype(np.float64)
    return X, y


def main(args):
    pd_data = pd.read_csv(str(args.feature_vectors))
    data = pd_data.as_matrix()

    if not args.train_all:
        np.random.seed(args.np_random_state)
        np.random.shuffle(data)
    names = data[:, 0]
    data = data[:, 1:]
    X, y = split_Xy(data, args.fv_length)

    if not args.train_all:
        split = round(X.shape[0] * args.train_split)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        test_names = names[split:]
    else:
        X_train, y_train = X, y

    print(f'Train with {X_train.shape[0]} samples, ' +
          f'each with {X_train.shape[1]} features.')


    classifier = get_classifier(args)

    print('>> Training classifier...', end=' ')
    classifier.fit(X_train, y_train)
    print('Done.')

    if not args.train_all:
        print('>> Predicting data...', end=' ')
        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f'Done. Test data mean accuracy: {score}')

        pred_filename = args.model_filename or classifier.default_filename
        pred_path = args.output_dir / f'{pred_filename}_results.csv'
        write_csv_predictions_vs_ground_truth(
            csv_path=pred_path,
            names=test_names,
            y_pred=y_pred,
            y=y_test,
        )
    print(f'>> Saved results to {pred_path}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = classifier.get_model_filename(args.model_filename)
    output_path = str(args.output_dir / filename)
    classifier.save(output_path)
    print(f'>> Saved model to {output_path}')


if __name__ == '__main__':
    main(collect_arguments())
