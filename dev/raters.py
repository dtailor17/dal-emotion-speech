"""
Raters class to encapsulate AL/DAL
"""

import csv
import collections
import random

from classifier import SVM

"""
Learning processes:
    - 'al' = Active Learning
    - 'dal' = Dynamic Active Learning

'learning_proc'='dal'
---------------------
'ordered'=False => 'rDAL'
'ordered'=True, 'order_updated'=False => 'oDAL' with fixed reliability values
'ordered'=True, 'order_updated'=True => 'oDAL' with updated reliability values
"""

class Raters:
    def __init__(self, data_test, learning_proc='dal', agreement_lvl=3, ordered=False, order_updated=False):
        self._learning_proc = learning_proc # Types: al, rdal, odal
        self._agreement_lvl = agreement_lvl
        self._ordered = ordered
        self._order_updated = order_updated
        self._data_test = data_test
        self._rater_labels = self.read_lbl_file('rater_labels/rater_labels.csv')
        self._n_raters = 5
        self._raters = [0,1,2,3,4]
        self._iter = 0

    def read_lbl_file(self, file_path):
        label_dict = {}
        for labels in csv.reader(open(file_path)):
            inst = labels[0]
            labels.pop(0)
            label_dict[inst] = [self.conv_to_int(lbl) for lbl in labels]
        return label_dict

    @staticmethod
    def conv_to_int(lbl):
        if lbl == 'IDL':
            return 0
        else:
            return 1

    def query(self, instances):
        self._iter += 1

        iter_rater_labels = [None for _ in range(len(instances.inst_names))]
        n_annotations = 0
        i = 0
        for inst_name in instances.inst_names:
            instances.y[i], n_queries, rater_lbls = self._get_label(inst_name)
            n_annotations += n_queries
            iter_rater_labels[i] = rater_lbls
            i += 1

        if self._learning_proc == 'dal' and self._ordered:
            if (self._iter == 1) or (self._order_updated and self._iter % 5 == 0):

                uar_scores = [0 for _ in range(self._n_raters)]

                for i in self._raters:
                    y_rater = [x[i] for x in iter_rater_labels]
                    rater_instances = instances.new_instance_set(range(len(instances.y)), y_rater)
                    svm_cls = SVM(prob_enabled=False, norm_type='std', resample_type='cls-wgt')
                    svm_cls.train(rater_instances)
                    uar = svm_cls.score('uar', self._data_test)
                    uar_scores[i] = (i, uar)

                uar_scores = sorted(uar_scores, key=lambda x: x[1])
                self._raters = [tup[0] for tup in uar_scores]

        return n_annotations

    def active_learning(self, inst_name):
        labels = [None for _ in range(self._n_raters)]
        for i in self._raters:
            labels[i] = self._request_label(i, inst_name)

        data_count = collections.Counter(labels)
        majority_lbl = data_count.most_common(1)[0][0]

        return majority_lbl, labels

    def dynamic_active_learning(self, inst_name):
        n_queries = 0
        labels = [None for _ in range(self._agreement_lvl)]
        j = self._agreement_lvl
        init_raters = self._raters[:j]
        spare_raters = self._raters[j:]

        i = 0
        for rater_id in init_raters:
            labels[i] = self._request_label(rater_id, inst_name)
            i += 1
        n_queries += j

        data_count = collections.Counter(labels)
        majority_lbl_count = data_count.most_common(1)[0]
        n_votes = majority_lbl_count[1]

        while n_votes < j:
            rater_id = spare_raters[0]
            spare_raters.pop()
            lbl = self._request_label(rater_id, inst_name)
            labels = labels + [lbl]
            data_count = collections.Counter(labels)
            majority_lbl_count = data_count.most_common(1)[0]
            n_votes = majority_lbl_count[1]
            n_queries += 1

        lbl_dal = majority_lbl_count[0]

        return (lbl_dal, n_queries)

    def _get_label(self, inst_name):
        # AL (Majoity Vote)
        if self._learning_proc == 'al':
            (label,_) = self.active_learning(inst_name)
            n_queries = self._n_raters
            return (label, n_queries, None)

        # DAL
        elif self._learning_proc == 'dal':
            if self._ordered:
                if (self._iter == 1) or (self._order_updated and self._iter % 5 == 0):
                    label, rater_lbls = self.active_learning(inst_name)
                    n_queries = self._n_raters
                    return (label, n_queries, rater_lbls)
                else:
                    (label, n_queries) = self.dynamic_active_learning(inst_name)
                    return (label, n_queries, None)
            else:
                random.shuffle(self._raters)
                (label, n_queries) = self.dynamic_active_learning(inst_name)
                return (label, n_queries, None)

    def _request_label(self, rater_id, inst_name):
        return self._rater_labels[inst_name][rater_id]
