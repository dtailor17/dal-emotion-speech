"""
BaseClassifier specifies methods required for learning process.
SVM is an implementation used in experiments of AL/DAL. It is highly
configurable with options for:
    - complexity value
    - confidence measure: Platt probability or distance-to-hyperplane
    - normalisation type: Z-std, Min-max
    - resampling type: over, under, class weight, none

Interface: BaseClassifier
Concrete Implementation: SVM
"""

import random
import operator
import heapq
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn import svm, metrics, preprocessing
from unbalanced_dataset.over_sampling import RandomOverSampler
from unbalanced_dataset.under_sampling import RandomUnderSampler

class BaseClassifier(object):
    __metaclass__  = ABCMeta

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def score(self, score_type, data):
        pass

    @abstractmethod
    def get_instance_subset(self, conf_type, n_inst, data):
        pass


class SVM(BaseClassifier):
    def __init__(self, complexity=0.5, prob_enabled=True, norm_type='std', resample_type='none'):
        self._complexity = complexity
        self._prob_enabled = prob_enabled

        if norm_type in ['std', 'minmax']:
            self._norm_type = norm_type
        else:
            raise ValueError('Unrecognised normalisation type ' + "'" + norm_type + "'")

        if resample_type in ['none', 'cls-wgt', 'over', 'under']:
            self._resample_type = resample_type
        else:
            raise ValueError('Unrecognised resampling type ' + "'" + resample_type + "'")

        # Random number seed required for random over-/under-sampling
        self._resample_seed = random.randint(0, 4294967295)
        # self._resample_seed = 1

    def train(self, data):
        self._cls = svm.SVC(
            C=self._complexity,
            kernel='linear',
            probability=self._prob_enabled
        )

        # Data unchanged if resample type 'none' or 'cls-wgt'
        X_resample = data.X
        y_resample = data.y

        if self._resample_type == 'cls-wgt':
            self._cls.class_weight = 'balanced'
        elif self._resample_type == 'over':
            resample = RandomOverSampler(ratio='auto', random_state=self._resample_seed)
            resample.verbose = False # Req. due to mistake in library
            X_resample, y_resample = resample.fit_transform(data.X, data.y)
        elif self._resample_type == 'under':
            resample = RandomUnderSampler(ratio='auto', random_state=self._resample_seed, verbose=False)
            X_resample, y_resample = resample.fit_transform(data.X, data.y)

        if self._norm_type == 'std':
            self._scaler = preprocessing.StandardScaler().fit(X_resample)
        else:
            self._scaler = preprocessing.MinMaxScaler().fit(X_resample)

        X_norm = self._scaler.transform(X_resample)

        self._cls.fit(X_norm, y_resample)

    def score(self, score_type, data):
        X_norm = self._scaler.transform(data.X)
        y_pred = self._cls.predict(X_norm)

        # print metrics.classification_report(data.y, y_pred)

        if score_type == 'uar':
            return self.uar(data.y, y_pred)
        else:
            raise ValueError('Unexpected score type ' + "'" + score_type + "'")

    def get_instance_subset(self, conf_type, n_inst, data):
        # conf_type: 'low, 'med', 'high'
        data_norm = self._scaler.transform(data.X)

        if self._prob_enabled:
            prob_class = self._cls.predict_proba(data_norm)
            # List of (index into dataset, (predicted label, probability of prediction) )
            prob_pred = [(i, max(enumerate(prob_class[i]), key=operator.itemgetter(1))) for i in range(len(prob_class))]
            # Map class labels list index to class labels
            target_lbls = self._cls.classes_
            conf_pred = [(idx, (target_lbls[pred_idx], prob)) for (idx, (pred_idx,prob)) in prob_pred]
        else:
            y_pred = self._cls.predict(data_norm)
            dist_to_plane = self._cls.decision_function(data_norm)
            # List of (index into dataset, (predicted label, absolute distance to hyperplane) )
            conf_pred = [(i,(y_pred[i], abs(dist_to_plane[i]))) for i in range(len(y_pred))]

        # Sort in descending order
        conf_pred.sort(key=lambda tup: tup[1][1], reverse=True)

        if conf_type == 'high':
            conf_pred = conf_pred[:n_inst]
        elif conf_type == 'med':
            med = np.median([conf for (_,(_,conf)) in conf_pred])
            conf_pred = heapq.nsmallest(n_inst, conf_pred, key=lambda x: abs(x[1][1]-med))
        elif conf_type == 'low':
            conf_pred = conf_pred[-n_inst:]
        else:
            raise ValueError('Unexpected confidence type ' + "'" + conf_type + "'")

        # Creates instances object to hold subset X and model predictions
        inst_indices = [idx for (idx,_) in conf_pred]
        y_pred = [pred for (_,(pred,_)) in conf_pred]
        data_subset = data.new_instance_set(inst_indices, y_pred)

        return data_subset

    @staticmethod
    def uar(y_truth, y_pred):
        recall_vals = metrics.recall_score(y_truth, y_pred, average=None)
        num_classes = len(set(y_truth))
        return sum(recall_vals) / float(num_classes)
