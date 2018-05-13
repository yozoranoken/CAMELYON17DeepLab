from abc import ABC
from abc import abstractmethod
from enum import Enum
import pickle

import numpy as np
import pandas as pd

from core import WSILabels
from gcforest import gcforest
from gcforest.utils.config_utils import load_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib



class Classifier(ABC):
    class Method(Enum):
        RANDOM_FOREST = 'random_forest'
        HDBSCAN = 'hdbscan'
        GC_FOREST = 'gc_forest'

    def __init__(self, args=None, model_path=None):
        if model_path is not None:
            self._clf = self._load(model_path)
        else:
            self._clf = self._make_clf(args)

    @property
    @abstractmethod
    def default_filename(self):
        pass

    @property
    def _extension(self):
        return 'pkl'

    def get_model_filename(self, filename):
        return f'{filename or self.default_filename}.{self._extension}'

    @abstractmethod
    def _load(self, model_path):
        pass

    @abstractmethod
    def _make_clf(self, args):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class RandomForest(Classifier):

    _MAX_FEATURES_VALS = 'auto', 'sqrt', 'log2'
    DEFAULT_MAX_FEATURE = _MAX_FEATURES_VALS[0]

    @classmethod
    def get_max_features_arg(cls, arg):
        val = arg
        if val is not None:
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except ValueError:
                if val not in cls._MAX_FEATURES_VALS:
                    raise ValueError('max_features not an int, float or in ' +
                                     f'{cls._MAX_FEATURES_VALS}')

        return val

    @property
    def default_filename(self):
        return 'rf_model'

    def _make_clf(self, args):
        return RandomForestClassifier(
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            verbose=args.verbose,
        )

    def _load(self, model_path):
        return joblib.load(str(model_path))

    def fit(self, X, y):
        return self._clf.fit(X, y)

    def save(self, path):
        return joblib.dump(self._clf, path)

    def predict(self, X):
        return self._clf.predict(X)


class GCForest(Classifier):

    @property
    def default_filename(self):
        return 'gc_model'

    def _make_clf(self, args):
        config = load_json(str(args.config))
        return gcforest.GCForest(config)

    def _load(self, model_path):
        with open(str(model_path), "rb") as pkl_file:
            return pickle.load(pkl_file)

    def _transform_X(self, X):
        X_4d = np.expand_dims(X, 1)
        X_4d = np.expand_dims(X_4d, 1)
        return X_4d

    def fit(self, X, y):
        return self._clf.fit_transform(self._transform_X(X), y)

    def save(self, path):
        with open(str(path), 'wb') as pkl_file:
            pickle.dump(self._clf, pkl_file, pickle.HIGHEST_PROTOCOL)

    def predict(self, X):
        return self._clf.predict(self._transform_X(X))


_CLF_MAP = {
    Classifier.Method.RANDOM_FOREST: RandomForest,
    Classifier.Method.GC_FOREST: GCForest,
}


def get_classifier(args=None, model_path=None):
    assert args is not None or model_path is not None
    return _CLF_MAP[Classifier.Method(args.method)](args, model_path)

def _convert_labels_str(labels):
    return tuple(map(lambda y: WSILabels(int(y)).name.lower(), labels))

_PRED_VS_GT_COL_NAMES = 'patient', 'prediction', 'label'
def write_csv_predictions_vs_ground_truth(csv_path, names, y_pred, y):
    stacked = np.column_stack((
        names,
        _convert_labels_str(y_pred),
        _convert_labels_str(y),
    ))
    data = pd.DataFrame(stacked, columns=_PRED_VS_GT_COL_NAMES)
    data.to_csv(str(csv_path), index=False)
