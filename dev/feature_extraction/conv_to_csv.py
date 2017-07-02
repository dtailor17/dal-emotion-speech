import os
import csv

script_dir = os.path.dirname(os.path.realpath(__file__))
corpus_lbl_path = os.path.join(script_dir, "labels/chunk_labels_2cl_corpus.txt")

train_path = os.path.join(script_dir, "labels/labels_train.csv")
test_path = os.path.join(script_dir, "labels/labels_test.csv")

w_train = csv.writer(open(train_path, 'w'))
w_test = csv.writer(open(test_path, 'w'))

with open(corpus_lbl_path) as f:
    for line in f:
        inst = line.split()
        name = inst[0]
        lbl = inst[1]
        if name.startswith("Ohm"):
            w = w_train
        else:
            w = w_test
        w.writerow([name, lbl])
