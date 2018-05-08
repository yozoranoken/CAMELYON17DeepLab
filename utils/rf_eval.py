#! /usr/bin/env python3.6
from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


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
    df = pd.read_csv(str(args.feature_vectors), header=0)
    print(df)


if __name__ == '__main__':
    main(collect_arguments())
