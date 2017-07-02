"""
Self-training using sklearn's svm.SVC classifier
"""

from instances import Instances
from classifier import SVM


if __name__ == '__main__':
    # Load datasets
    data_train = Instances()
    data_train.load_from_file('dataset/train_200/train_eGeMAPS.arff')

    data_unlabelled = Instances()
    data_unlabelled.load_from_file('dataset/train_200/unlabelled_eGeMAPS.arff')

    data_test = Instances()
    data_test.load_from_file('feature_extraction/arff/aibo_test_eGeMAPS.arff')

    # Initialise SVM classifier with particular configuration
    complexity = 0.07432544468767006
    svm_cls = SVM(complexity=complexity, prob_enabled=True, norm_type='std', resample_type='over')
    svm_cls.train(data_train) # Initial train

    ssl_iterations = 25
    num_instances_to_label = 200

    uar = svm_cls.score('uar', data_test) # Performance score before ST
    print uar

    # Self-training loop
    for i in range(1, ssl_iterations+1):
        # Get instances with high confidence from unlabelled dataset
        data_high_conf = svm_cls.get_instance_subset('high', num_instances_to_label, data_unlabelled)

        data_train.add_instances(data_high_conf) # Add to labelled set
        data_unlabelled.remove_instances(data_high_conf) # Remove from unlabelled set

        svm_cls.train(data_train) # Re-train
        uar = svm_cls.score('uar', data_test) # Updated performance score
        print uar
