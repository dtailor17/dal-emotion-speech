"""
Self-training using WEKA SMO classifier
"""

import random
import weka.core.jvm as jvm
from pandas import DataFrame
from weka.core.dataset import Instances
from weka.core.dataset import InstanceIterator
from weka.filters import Filter
from weka.classifiers import Classifier, PredictionOutput, Evaluation
from util.partition_dataset import _remove_instances as remove_instances
from util.partition_dataset import _load_arff as load_arff


def train_svm(train_data):
    """
    Trains SMO (WEKA) using same parameters as 'Z. Zhang - Cooperative Learning' paper
    http://weka.sourceforge.net/doc.dev/weka/classifiers/functions/SMO.html
    """
    svm_options = [
        "-C", "0.05",
        "-L", "0.001",
        "-P", "1.0E-12",
        "-N", "0",
        "-V", "-1",
        "-W", "1",
        "-M",
        "-K", "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007",
        "-calibrator", "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4"
    ]

    cls_svm = Classifier(classname="weka.classifiers.functions.SMO", options=svm_options)
    cls_svm.build_classifier(remove_string_attr(train_data))
    return cls_svm


def transform_pred_data_pandas(data):
    """
    Converts WEKA's prediction output to Pandas Dataframe
    """
    # List containing each instance
    pred = [line.split() for line in data.splitlines()]

    # If last entry is empty list, delete it
    if not pred[-1]:
        del pred[-1]

    # Split each instance entry into individual data items
    max_len = len(max(pred, key=len))
    if max_len == 5: # Data contained class labels
        for instance in pred:
            if len(instance) < max_len:
                instance.insert(-1, '')
        columns = ['inst#', 'actual', 'predicted', 'error', 'probability']
    else:
        columns = ['inst#', 'actual', 'predicted', 'probability']

    # Convert to Pandas dataframe
    df = DataFrame(pred, columns=columns)
    return df


def get_predictions(cls_svm, data_arff):
    """
    Predictions on test set with probability
    https://weka.wikispaces.com/Making+predictions
    """
    pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
    pout.print_classifications(cls_svm, remove_string_attr(data_arff))
    pred_data = pout.buffer_content()

    return transform_pred_data_pandas(pred_data)


def generate_rand_seed():
    """
    Generates random number between Java's minimum and maximum sized integer
    """
    return random.randint(-2147483648, 2147483647) # Java max/min integer


def upsample(data_arff, seed=1, size_percentage=200.0):
    """
    Resamples dataset with an equal class distribution
    """
    options = [
        "-B", "1.0",
        "-S", str(seed),
        "-Z", str(size_percentage)
    ]

    resample_filter = Filter(classname="weka.filters.supervised.instance.Resample", options=options)
    resample_filter.inputformat(data_arff)
    resample_arff = resample_filter.filter(data_arff)

    return resample_arff

def evaluate_classifier(classifier, data_test):
    """
    Generates dictionary of performance metrics from evaluating classifier on
    given test set
    http://pythonhosted.org/python-weka-wrapper/weka.html#weka.classifiers.Evaluation
    """
    eval = Evaluation(remove_string_attr(data_test))
    eval.test_model(classifier, remove_string_attr(data_test))

    metrics = {}
    metrics['weighted_recall'] = eval.weighted_recall

    num_lbls = data_test.class_attribute.num_values
    metrics['number_labels'] = num_lbls

    recall_vals = [0 for _ in range(num_lbls)]
    for i in range(num_lbls):
        metrics[i] = eval.recall(i)
        recall_vals[i] = metrics[i]

    metrics['unweighted_recall'] = sum(recall_vals)/float(num_lbls)

    return metrics


def get_uar(cls_svm, data_test):
    """
    Returns Unweighted Average Recall (UAR)
    """
    metrics = evaluate_classifier(cls_svm, remove_string_attr(data_test))
    return metrics['unweighted_recall']


def remove_string_attr(data_arff):
    """
    Remove all attributes with string type
    """
    arff_copy = Instances.copy_instances(data_arff)
    arff_copy.delete_attribute_type(2)
    return arff_copy


def perform_self_training(cls_svm, data_train, data_unlabelled, num_instances=500):
    """
    Executes single iteration of self-training
    """
    # Generate prediction/confidence table then choose 500 instances
    # with highest confidence
    df = get_predictions(cls_svm, data_unlabelled)
    df_sorted = df.sort_values(by=['probability'], ascending=[False])
    df_ssl = df_sorted.iloc[range(num_instances)]
    inst_pred = zip(df_ssl['inst#'], df_ssl['predicted'])
    lowest_confidence = df_ssl.iloc[num_instances-1]['probability']

    # New dataset to hold selected instances
    data_ssl = Instances.template_instances(data_unlabelled)

    # Add selected instances to data_ssl
    inst_nums = [int(x[0]) for x in inst_pred]
    for inst_num in inst_nums:
        inst = data_unlabelled.get_instance(inst_num-1) # 0-based
        data_ssl.add_instance(inst)

    remove_instances(data_unlabelled, data_ssl)

    # Add class labels to data_ssl
    class_lbls = [x[1] for x in inst_pred]
    i = 0
    for instance in InstanceIterator(data_ssl):
        instance.set_value(data_ssl.class_index, float(class_lbls[i][0])-1.0)
        i += 1

    # Add data_ssl to data_train
    # Switched argument order to fix IndexOutOfBoundsException
    return (Instances.append_instances(data_ssl, data_train), lowest_confidence)


if __name__ == '__main__':
    jvm.start()

    seed = generate_rand_seed()

    data_train = load_arff("dataset/train_200/train_IS09.arff")
    data_unlabelled = load_arff("dataset/train_200/unlabelled_IS09.arff")
    data_test = load_arff("feature_extraction/arff/aibo_test_IS09.arff")

    print "\nIteration 0"
    print "Number of machine-labelled instances:", 0

    cls_svm = train_svm(upsample(data_train, seed))
    uar = get_uar(cls_svm, data_test)
    print "UAR:", uar, '\n'

    ssl_iterations = 25
    num_instances_to_label = 200

    inst_uar_data = [0 for _ in range(ssl_iterations+1)]
    inst_uar_data[0] = (0, uar)

    for i in range(1, ssl_iterations+1):
        (data_train, min_confidence) = \
            perform_self_training(cls_svm,
                                  data_train, data_unlabelled, num_instances_to_label)
        cls_svm = train_svm(upsample(data_train, seed))

        num_instances = num_instances_to_label*i
        uar = get_uar(cls_svm, data_test)

        print "Iteration", i
        print "Number of machine-labelled instances:", num_instances
        print "Confidence lower bound:", min_confidence
        print "UAR:", uar, '\n'

        inst_uar_data[i] = (num_instances, uar)

    print "Unlabelled instances remaining:", data_unlabelled.num_instances, '\n'

    print "X-data:"
    print [x[0] for x in inst_uar_data], '\n'
    print "Y-data:"
    print [x[1] for x in inst_uar_data]

    jvm.stop()
