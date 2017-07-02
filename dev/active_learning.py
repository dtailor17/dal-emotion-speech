"""
Active-training using sklearn's svm.SVC classifier
"""

from instances import Instances
from classifier import SVM
from raters import Raters


if __name__ == '__main__':
    # Load datasets
    data_train = Instances()
    data_train.load_from_file('dataset/train_200/train_eGeMAPS.arff')

    data_unlabelled = Instances()
    data_unlabelled.load_from_file('dataset/train_200/unlabelled_eGeMAPS.arff')

    data_test = Instances()
    data_test.load_from_file('feature_extraction/arff/aibo_test_eGeMAPS.arff')

    # raters = Raters(data_test=data_test, learning_proc='dal', agreement_lvl=3, ordered=True, order_updated=True)
    raters = Raters(data_test=data_test, learning_proc='al')

    # Initialise SVM classifier with particular configuration
    complexity = 0.07432544468767006
    svm_cls = SVM(complexity=complexity, prob_enabled=True, norm_type='std', resample_type='over')
    svm_cls.train(data_train)

    ssl_iterations = 25
    num_instances_to_label = 200

    uar = svm_cls.score('uar', data_test) # Performance score before AL
    n_annotations = 0

    uar_values = []
    annotation_count_vals = []

    # print uar
    # print n_annotations
    # print
    uar_values.append(uar)
    annotation_count_vals.append(n_annotations)

    # Active learning loop
    for i in range(1, ssl_iterations+1):
        # Get instances with low/medium confidence from unlabelled set
        data_ssl = svm_cls.get_instance_subset('low', num_instances_to_label, data_unlabelled)

        n_annotations += raters.query(data_ssl) # Updates data_ssl with rater labels

        data_train.add_instances(data_ssl) # Add to labelled set
        data_unlabelled.remove_instances(data_ssl) # Remove from unlabelled set

        svm_cls.train(data_train) # Re-trains
        uar = svm_cls.score('uar', data_test) # Updated performance score

        # print uar
        # print n_annotations
        # print

        uar_values.append(uar)
        annotation_count_vals.append(n_annotations)

    # print
    # for uar in uar_values:
    #     print uar
    #
    # print
    # for annot_count in annotation_count_vals:
    #     print annot_count
