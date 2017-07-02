"""
Representation for dataset where:
    - matrix 'X' (each row is instance)
    - list of integers 'y'
    - list of strings 'inst_names'

Interface: BaseInstances
Concrete Implementation: Instances
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from util.arff_util import load_arff


class BaseInstances(object):
    __metaclass__  = ABCMeta

    @abstractmethod
    def __init__(self, X=None, y=None, inst_names=None):
        self.X = X
        self.y = y
        self.inst_names = inst_names

    @abstractmethod
    def load_from_file(self, file_path):
        pass

    @abstractmethod
    def new_instance_set(self, inst_indices, y_pred):
        pass

    @abstractmethod
    def add_instances(self, data):
        pass

    @abstractmethod
    def remove_instances(self, data):
        pass


class Instances(BaseInstances):
    def __init__(self, X=None, y=None, inst_names=[]):
        super(Instances, self).__init__(X=X,
                                        y=y,
                                        inst_names=inst_names)

    def load_from_file(self, file_path):
        (self.X, self.y, self.inst_names) = load_arff(file_path)

    def new_instance_set(self, inst_indices, y_pred):
        new_instances = Instances()
        new_instances.X = self.X[inst_indices,:]
        new_instances.y = y_pred
        new_instances.inst_names = [self.inst_names[idx] for idx in inst_indices]
        return new_instances

    def add_instances(self, data):
        if self.X is not None:
            self.X = np.concatenate([self.X, data.X])
            self.y = np.concatenate([self.y, data.y])
        else:
            self.X = data.X
            self.y = data.y

        self.inst_names = np.concatenate([self.inst_names, data.inst_names])

    def remove_instances(self, data):
        # Flatten matrices (1 column)
        inst_orig = self.X
        inst_smp = data.X
        orig_cond = inst_orig.view([('', inst_orig.dtype)] * inst_orig.shape[1])
        smp_cond = inst_smp.view([('', inst_smp.dtype)] * inst_smp.shape[1])

        # Get indices into original X of instances found in sample set
        idx_to_del = [0 for _ in range(len(smp_cond))]
        i = 0
        for inst in smp_cond:
            idx_to_del[i] = np.where(orig_cond == inst)[0][0]
            i += 1

        self.X = np.delete(self.X, idx_to_del, axis=0)
        self.y = np.delete(self.y, idx_to_del, axis=0)
        self.inst_names = np.delete(self.inst_names, idx_to_del, axis=0)
