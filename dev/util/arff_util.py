"""
Called by class Instances from instances.py to load ARFF as (X, y, inst_names)
"""

import arff
import numpy as np


def load_arff(file_path):
    with open(file_path) as f:
        # Load arff as dict with keys: (attributes, relation, description, data)
        dataset = arff.load(f, encode_nominal=True)
        n_inst = len(dataset['data'])

        # Instance names (i.e. wav filenames)
        inst_names = ['' for _ in range(n_inst)]
        # Target values encoded (IDL=>0, NEG=>1, ?=>None)
        y = [None for _ in range(n_inst)]

        # Remove name and target from data (X)
        X = dataset['data']
        idx = 0
        for inst in X:
            inst_names[idx] = inst[0]
            y[idx] = inst[-1]
            del inst[0]
            del inst[-1]
            idx += 1

        # Convert to numpy 2darray for sklearn training
        X = np.asarray(X, dtype=np.float64)

        return (X,y,inst_names)


if __name__ == '__main__':
    arff_path = '../feature_extraction/arff/aibo_train_eGeMAPS.arff'
    (X, y, inst_names) = load_arff(arff_path)
