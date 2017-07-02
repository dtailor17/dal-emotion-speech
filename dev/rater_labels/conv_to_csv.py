import os
import csv

script_dir = os.path.dirname(os.path.realpath(__file__))
aibo_train_filenames_path = os.path.join(script_dir, "aibo_train_filenames.txt")
rater_labels_path = os.path.join(script_dir, "rater_labels.txt")

csv_path = os.path.join(script_dir, "rater_labels.csv")
csv_writer = csv.writer(open(csv_path, 'w'))

with open(aibo_train_filenames_path) as f:
    instance_names = f.read().splitlines()

with open(rater_labels_path) as f:
    labels = f.read().splitlines()

for i in range(len(instance_names)):
    name = instance_names[i]
    rater_lbls = labels[i].split()
    rater_lbls[0] = name
    csv_writer.writerow(rater_lbls)
