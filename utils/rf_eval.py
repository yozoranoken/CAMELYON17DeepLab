#! /usr/bin/env python3.6
from argparse import ArgumentParser
from collections import OrderedDict
import csv
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from core import WSILabels

_EVAL_DIRNAME = 'eval'


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

    parser.add_argument(
        '--exclude-label-col',
        action='store_true',
    )


    return parser.parse_args()


class pNStage(Enum):
    PN0 = 'pN0'
    PN0_I_PLUS = 'pN0(i+)'
    PN1_MI = 'pN1mi'
    PN1 = 'pN1'
    PN2 = 'pN2'


def get_pNstage(slide_predictions):
    non_negative_count = sum(1 for pred in slide_predictions
                             if pred != WSILabels.NEGATIVE)
    if non_negative_count > 0:
        if WSILabels.MACRO in slide_predictions:
            if non_negative_count > 3:
                stage = pNStage.PN2
            else:
                stage = pNStage.PN1
        elif WSILabels.MICRO in slide_predictions:
            stage = pNStage.PN1_MI
        else:
            stage = pNStage.PN0_I_PLUS
    else:
        stage = pNStage.PN0

    return stage


def aggregate_for_pNstage_results(slide_names, predictions):
    assert slide_names.shape[0] == predictions.shape[0]

    slide_predictions = list(zip(slide_names, predictions))
    slide_predictions.sort(key=lambda x: x[0])

    patients = OrderedDict()
    for slide_name, prediction in slide_predictions:
        patient_name = slide_name[:11] + '.zip'
        patient = patients.get(patient_name, OrderedDict())

        if not patient:
            patients[patient_name] = patient

        patient[slide_name + '.tif'] = WSILabels(prediction)

    results = OrderedDict()
    for patient, slides in patients.items():
        stage_label = get_pNstage(slides.values())
        results[patient] = stage_label.value
        for slide_name, label in slides.items():
            results[slide_name] = label.name.lower()

    return results


def main(args):
    data = pd.read_csv(str(args.feature_vectors), header=0).as_matrix()
    names = data[:, 0]

    X = data[:, 1:].astype(np.float64)
    if args.exclude_label_col:
        X = X[:, :-1]

    rf_classifier = joblib.load(str(args.model_pkl))
    y = rf_classifier.predict(X)

    pNstage_results = aggregate_for_pNstage_results(names, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / args.filename
    with open(str(output_file), 'w') as pred_file:
        writer = csv.writer(pred_file)
        writer.writerow(['patient', 'stage'])
        for patient, stage in pNstage_results.items():
            writer.writerow([patient, stage])

    print(f'>> Predictions written at {output_file}')


if __name__ == '__main__':
    main(collect_arguments())
