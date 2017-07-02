"""
Script to convert ARFF to LIBSVM for use in scikit-learn
(No longer possible since newly generated ARFF files have string attribute -
wav file name - which is not supported)
"""

import weka.core.jvm as jvm
from weka.core import converters


def arff_to_libsvm(arff_path, libsvm_path):
    """
    Load instances from given ARFF file and save as LIBSVM file
    """
    loader = converters.Loader(classname='weka.core.converters.ArffLoader')
    data = loader.load_file(arff_path)
    saver = converters.Saver(classname='weka.core.converters.LibSVMSaver')
    saver.save_file(data, libsvm_path)


if __name__ == '__main__':
    jvm.start()

    # TODO: Use argparse (command line arguments)
    in_arff = '../feature_extraction/arff/aibo_train_IS09.arff'
    out_libsvm = '../dataset/test.libsvm'
    arff_to_libsvm(in_arff, out_libsvm)

    jvm.stop()
