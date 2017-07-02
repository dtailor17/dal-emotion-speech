#!/usr/bin/env bash

# Get directory containing bash script
WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Uncompress files
cat aibo_split.tgz_* | tar xz

# Compress and split files
# tar cz aibo_test_IS09.arff aibo_test_IS13.arff aibo_test_eGeMAPS.arff aibo_train_IS09.arff aibo_train_IS13.arff aibo_train_eGeMAPS.arff | split -b 95mB - aibo_split.tgz_
