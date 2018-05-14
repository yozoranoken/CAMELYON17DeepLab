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
from ensembles import Classifier
from ensembles import get_classifier
from ensembles import split_Xy
from sklearn.metrics import accuracy_score

_EVAL_DIRNAME = 'eval'


def collect_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        '--feature-vectors',
        type=Path,
        required=True
    )

    parser.add_argument(
        '--fv-length',
        type=int,
        required=True
    )

    parser.add_argument(
        '--method',
        choices=tuple(method for method in Classifier.Method),
        type=Classifier.Method,
        required=True,
    )

    parser.add_argument(
        '--model-path',
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
                             if pred not in (WSILabels.NEGATIVE,
                                             WSILabels.ITC))
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

    return np.array([(patient, stage) for patient, stage in results.items()])


def main(args):
    pd_data = pd.read_csv(str(args.feature_vectors))
    data = pd_data.as_matrix()

    names = data[:, 0]
    data = data[:, 1:]
    X, y = split_Xy(data, args.fv_length)

    classifier = get_classifier(args.method, model_path=args.model_path)

    print('>> Predicting data...', end=' ')
    y_pred = classifier.predict(X)
    score = accuracy_score(y, y_pred)
    print(f'Done. Test data mean accuracy: {score}')

    # pred_filename = args.model_filename or classifier.default_filename
    # pred_path = args.output_dir / f'{pred_filename}_results.csv'
    # write_csv_predictions_vs_ground_truth(
    #     csv_path=pred_path,
    #     names=test_names,
    #     y_pred=y_pred,
    #     y=y_test,
    # )
    # print(f'>> Saved results to {pred_path}')

    pNstage_results = aggregate_for_pNstage_results(names, y_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.filename or f'{args.model_path.stem}_predictions.csv'
    output_file = args.output_dir / filename

    data = pd.DataFrame(pNstage_results, columns=('patient', 'stage'))
    data.to_csv(str(output_file), index=False)

    print(f'>> Predictions written at {output_file}')


if __name__ == '__main__':
    main(collect_arguments())
