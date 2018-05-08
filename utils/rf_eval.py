#! /usr/bin/env python3.6
from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from core import WSILabels


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--feature-vectors',
        type=Path,
        required=True
    )

    parser.add_argument(
        '--model-pkl',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--filename',
        type=str,
        default='rf_predictions.csv'
    )


    return parser.parse_args()


def main(args):
    data = pd.read_csv(str(args.feature_vectors), header=0).as_matrix()
    names = data[:, 0]
    X = data[:, 1:].astype(np.float64)

    rf_classifier = joblib.load(str(args.model_pkl))
    y = rf_classifier.predict(X)
    y = tuple(map(lambda val: WSILabels(val).name.lower(), y))

    with open(str(args.output_dir / args.filename), 'w') as pred_file:
        writer = csv.writer(pred_file)
        writer.writerow(['uid', 'prediction'])
        for n, pred in zip(names, y):
            writer.writerow([n, pred])


if __name__ == '__main__':
    main(collect_arguments())
