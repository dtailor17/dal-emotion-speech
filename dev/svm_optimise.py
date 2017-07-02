"""
Optimise hyperparameters for SVM from sklearn
Aibo training set
Cross validation
"""

import math
import random
import numpy as np
from unbalanced_dataset.over_sampling import RandomOverSampler
from sklearn import cross_validation, datasets, grid_search, metrics, preprocessing, svm

from util.arff_util import load_arff


def uar(y_truth, y_pred):
    """
    Computes unweighted average recall (UAR)
    """
    recall_vals = metrics.recall_score(y_truth, y_pred, average=None)
    num_classes = len(set(y_truth))
    return sum(recall_vals) / float(num_classes)


def get_hyper_params():
    # # Linear kernel
    # c_vals, step_size = np.linspace(
    #     start=math.pow(2,-12),
    #     stop=math.pow(2,12),
    #     num=50,
    #     retstep=True
    # )
    c_pow = range(-12,13,1)
    # c_pow = np.arange(-12, 13.25, 0.25)
    c_vals = [math.pow(2,i) for i in c_pow]
    params = {'C': c_vals, 'kernel': ['linear']}

    # # RBF kernel
    # c_vals = np.linspace(
    #     start=math.pow(2,-12),
    #     stop=math.pow(2,12),
    #     num=1000
    # )
    # g_vals = np.linspace(
    #     start=math.pow(2,-15),
    #     stop=math.pow(2,10),
    #     num=5000
    # )
    # params = {'C': c_vals, 'gamma': g_vals, 'kernel': ['rbf']}

    # Polynomial kernel
    # c_vals = np.linspace(
    #     start=math.pow(2,-12),
    #     stop=math.pow(2,13),
    #     num=1000
    # )
    # g_vals = np.linspace(
    #     start=math.pow(2,-15),
    #     stop=math.pow(2,11),
    #     num=1000
    # )
    # coef_vals = np.linspace(
    #     start=0.0,
    #     stop=1.0,
    #     num=1000
    # )
    # params = {
    #     'C': c_vals,
    #     'gamma': g_vals,
    #     'coef0': coef_vals,
    #     'degree': [3],
    #     'kernel': ['poly']
    # }

    return params, c_pow


# Custom scorer using uar
uar_score = metrics.make_scorer(uar, greater_is_better=True)

# Load data as sparse matrices
# X.shape (n_samples, n_features); y (label vector)
# X_train, y_train, _ = load_arff('feature_extraction/arff/aibo_train_IS09.arff')
X_train, y_train, _ = load_arff('dataset/train_200/train_IS09.arff')
X_test, y_test, _ = load_arff('feature_extraction/arff/aibo_test_IS09.arff')

# Initialise SVM estimator
svm_estimator = svm.SVC(class_weight=None, probability=True) # Adjusts C using class weight (i.e. resample)

# Oversampling
# seed = random.randint(0, 4294967295)
resample = RandomOverSampler(ratio='auto', random_state=1)
resample.verbose = False # Req. due to mistake in library
X_resample, y_resample = resample.fit_transform(X_train, y_train)

# Normalize data
scaler = preprocessing.StandardScaler().fit(X_resample) # Z-standardization
# scaler = preprocessing.MinMaxScaler().fit(X_resample)  # Min-max normalization
X_train_norm = scaler.transform(X_resample)
X_test_norm = scaler.transform(X_test)

# Hyperparameters for grid search optimisation
params, c_pow = get_hyper_params()

# Stratified 10-fold CV
cv_iter = cross_validation.StratifiedKFold(
    y_train,
    n_folds=10,
    shuffle=True,
    random_state=1
)

# Grid search
clf = grid_search.GridSearchCV(
    svm_estimator,
    param_grid=params,
    scoring=uar_score,
    cv=cv_iter,
    n_jobs=8, # Jobs run in parallel on all CPUs
    verbose=3
)

clf.fit(X_train_norm, y_resample)

# Display results
print
print 'UAR for all parameter combinations'
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
print

print 'Optimal parameters:', clf.best_params_
print 'UAR:', clf.best_score_
print

print 'Test set performance'
y_true, y_pred = y_test, clf.predict(X_test_norm)
print 'UAR:', uar(y_true, y_pred)
print metrics.classification_report(y_true, y_pred)

print
for x in c_pow:
    print x

print
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f" % (mean_score)
