"""
openSMILE feature extraction
"""

import os
import csv
import subprocess


def read_csv(file_path):
    label_dict = {}
    for key, val in csv.reader(open(file_path)):
        label_dict[key] = val
    return label_dict


script_dir = os.path.dirname(os.path.realpath(__file__))
# smile_path = "/homes/dvt13/private/openSMILE-2.1.0/bin/linux_x64_standalone_static/SMILExtract"
smile_path = "/Users/dharmeshtailor/openSMILE-2.1.0/inst/bin/SMILExtract"
# config_path = os.path.join(script_dir, "configs/IS09_emotion/IS09_emotion.conf")
# config_path = os.path.join(script_dir, "configs/eGeMAPS/eGeMAPSv01a.conf")
config_path = os.path.join(script_dir, "configs/IS13_ComParE/IS13_ComParE.conf")

arff_name = "aibo_test_IS13"
arff_path = os.path.join(script_dir, arff_name + ".arff")
lbl_path = os.path.join(script_dir, "labels/labels_test.csv")
wav_all_path = os.path.join(script_dir, "AIBO/test/")

label_dict = read_csv(lbl_path) # Dict of wav_name->label
FNULL = open(os.devnull, 'w') # Hide openSMILE output

# Delete ARFF if already exists
try:
    os.remove(arff_path)
except OSError:
    pass

for fn in os.listdir(wav_all_path):
    name = fn.split(".", 1)[0]
    wav_path = os.path.join(wav_all_path, fn)
    subprocess.call([
        smile_path,
        "-C", config_path,
        "-I", wav_path,
        "-O", arff_path,
        "-relation", arff_name,
        "-classlabel", label_dict[name],
        "-instname", name
    ], stdout=FNULL, stderr=subprocess.STDOUT)
