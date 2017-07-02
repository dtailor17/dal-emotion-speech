"""
Script to partition dataset (ARFF) into 'training' set and 'unlabelled' set
Number of instances in training set is specified
"""

import math
import arff
import weka.core.jvm as jvm
from weka.filters import Filter
from weka.core.converters import Loader
from weka.core.dataset import InstanceIterator


def _load_arff(path_arff):
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data_arff = loader.load_file(path_arff)
    data_arff.class_is_last()
    return data_arff


def _remove_instances(inst_init, inst_sample):
    # List containing every element of inst_init as a string
    inst_str_rep = ['' for _ in range(inst_init.num_instances)]
    i = 0
    for instance in InstanceIterator(inst_init):
        inst_str_rep[i] = str(instance)
        i += 1

    # Get indices into inst_init for which elements are to be deleted
    idx_lst = [0 for _ in range(inst_sample.num_instances)]
    i = 0
    for instance in InstanceIterator(inst_sample):
        # Returns 1st occurence of instance
        # TODO: What if 'inst_init' contains same instance 2+ times
        idx_lst[i] = inst_str_rep.index(str(instance))
        i += 1
    idx_set = set(idx_lst)

    # Delete instances
    for i in reversed(range(inst_init.num_instances)):
        if i in idx_set:
            inst_init.delete(i)


def _save_arff(data_arff, filename):
    """
    Saving arff files using weka-wrapper led to loss of precision in numeric
    values
    Work around: using liac-arff to construct 'arff_obj'
    """
    # List of tuples of form: (attribute-name, type/values)
    attr_info = [0 for _ in range(data_arff.num_attributes)]
    nominal_attrs_idx = [] # Indices of nominal attributes
    string_attrs_idx = [] # Indices of string attributes
    i = 0
    for attr in data_arff.attributes():
        attr_info[i] = (attr.name, _get_attr_type(attr))
        if attr.type == 1: # Nominal
            nominal_attrs_idx.append(i)
        elif attr.type == 2: # String
            string_attrs_idx.append(i)
        i += 1

    # List of instances values (as a list)
    data_info = [0 for _ in range(data_arff.num_instances)]
    i = 0
    for instance in InstanceIterator(data_arff):
        data_info[i] = instance.values.tolist()
        # 'instance.values' returns all values as float
        # Convert 'float' value of nominal attributes to string
        for idx in nominal_attrs_idx:
            val = data_info[i][idx]
            if math.isnan(val):
                data_info[i][idx] = '?'
            else:
                data_info[i][idx] = data_arff.attribute(idx).value(int(val))

        for idx in string_attrs_idx:
            data_info[i][idx] = instance.get_string_value(idx)

        i += 1

    # Construct liac-arff dataset
    arff_obj = {
        'description': u'',
        'relation': data_arff.relationname, # TODO: Change to something more appropriate
        'attributes': attr_info,
        'data': data_info
    }

    my_file = open(filename, 'w')
    arff.dump(arff_obj, my_file)
    my_file.close()


# TODO: Types not dealt with: date, relational
def _get_attr_type(attr):
    """
    http://weka.sourceforge.net/doc.dev/weka/core/Attribute.html
    """
    if attr.type == 0: # Numeric
        return 'REAL'
    elif attr.type == 1: # Nominal
        return attr.values
    elif attr.type == 2: # String
        return 'STRING'


def _get_sample(data_arff, sample_size):
    """
    Sample with same class distribution as original data
    http://weka.sourceforge.net/doc.dev/weka/filters/supervised/instance/Resample.html
    """
    if data_arff.num_instances <= sample_size:
        raise RuntimeError('Sample size greater than size of dataset')

    percentage_input_set = sample_size / (data_arff.num_instances * 1.0) * 100
    # percentage_input_set = 5.031 # Hack to get 500 (not 499)
    # percentage_input_set = 2.02 # Hack to get 200 (not 199)

    options = [
        "-S", "1", #str(generate_rand_seed())
        "-Z", str(percentage_input_set),
        "-no-replacement"
    ]

    resample_filter = Filter(classname="weka.filters.supervised.instance.Resample", options=options)
    resample_filter.inputformat(data_arff)
    sample_arff = resample_filter.filter(data_arff)

    return sample_arff


def train_unlabel_partition(dataset_path, train_path, unlabel_path, train_size):
    """
    Training / Unlabelled datasets partitioned from Aibo dataset
    required to start Self-Training process
    """
    data_unlabelled = _load_arff(dataset_path)

    data_train = _get_sample(data_unlabelled, train_size)
    _remove_instances(data_unlabelled, data_train)

    # Remove class label from instances in 'unlabelled' dataset
    class_idx = data_unlabelled.class_index
    for instance in InstanceIterator(data_unlabelled):
        instance.set_missing(class_idx)

    _save_arff(data_train, train_path)
    _save_arff(data_unlabelled, unlabel_path)


if __name__ == '__main__':
    jvm.start()

    # TODO: Use argparse (command line arguments)
    train_set_size = 500
    dataset_path = '../feature_extraction/arff/aibo_train_IS09.arff'
    train_path = '../dataset/train_200/train1.arff'
    unlabel_path = '../dataset/train_200/unlabelled1.arff'

    try:
        train_unlabel_partition(dataset_path, train_path, unlabel_path, train_set_size)
    finally:
        jvm.stop()
