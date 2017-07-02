# dal-emotion-speech
Final year project on Dynamic Active Learning for Emotional Speech Recognition

Virtual Environment
-------------------
    ./setup.sh
This sets up the virtual environment (ensure package `virtualenv` is installed) and installs all required packages, as listed in `requirements.txt`

    source venv/bin/activate
This activates our virtual environment's python distribution. Note for lab machines you need to run `active.csh`. Before executing any code, always ensure the correct python is being used: `/venv/bin/python`. This can be verified by executing: `which python`

    pip install <package-name>
Run this command, replacing `<package-name>` as appropriate, to install a new package. `pip freeze` displays the list of installed packages and should now show the newly installed package.

    rm -f requirements.txt
    pip freeze > requirements.txt
This adds the change to the project's list of dependencies.

Web Application (Incomplete)
----------------------------
    ./start.sh
This starts the Flask (Python) server

```
app
 |    
 +-- __init__.py
 +-- static
 |  |  
 |  +-- audio/speech01.wav
 |  +-- css/style.css
 |  +-- img
 |  |   |
 |  |   +-- cloud.png
 |  |   +-- favicon.ico
 |  +-- js/init.js
 |
 +-- templates
 |  |  
 |  +-- annotate.html
 |  +-- index.html
 |  +-- signup.html
```

HTML uses `materialize-css` for styling and UI components.
Flask is the (Python) back-end framework with routes specified in `app\__init__.py`

Feature Extraction
------------------
    dev/feature_extraction
`\arff` contains the training and testing datasets for each feature set

`\config` contains the configuration files used by openSMILE to perform the feature extraction

`\labels` contains the labels (IDL/NEG) for each `.wav` file

    ./extract.py
This performs the feature extraction. The following variables need to be set appropriately:
* smile_path
* config_path
* arff_name
* lbl_path
* wav_all_path


Labelled/Unlabelled Set Partition
-----------------------

    dev/util/partition_dataset.py
This performs the partition of the given training set into train and unlabelled using `python-weka-wrapper`.
The following variables need to be adjusted:
* train_set_size
* dataset_path
* train_path
* unlabel_path


    dev/dataset
This contains the ARFF files required for DAL.
There are two directories `train_200` and `train_500` in which the `train_*.arff` files contain the corresponding number of instances.
The unlabelled instances are stored in `unlabelled_*.arff`.
These datasets are read in `active_learning.py` and `self_training.py`.
`train_IS09.arff` and `train_IS13.arff` as well as the corresponding unlabelled datasets have been removed due to space limitations.

SVM Optimisation
----------------
    dev/svm_optimise.py
This runs the grid search using `scikit-learn` to determine best hyperparameters for classifier `svm.SVC`.
Optimal complexity parameter then used in `self_training.py` and `active_learning.py`.


Self-Training
--------------
    dev/self_training.py
This runs the self-training process using classes `Instances` from `instances.py` and `SVM` from `classifier.py`.

AL/DAL
------
    dev/active_learning.py
This runs the (dynamic) active learning process.
In addition to the classes above, this uses class `Rater` from `raters.py` which encapsulates the particular type of DAL.

    dev/rater_labels/rater_labels.csv
This specifies the audio file name along with the 5 annotations.
It is read at the start of the active learning process when `Raters` is initialised.

Results
-------
    /experiments/new_results
This contains the UAR performance data of the various configurations as presented in the final presentation in MATLAB format.
Plots were generated using `/experiments/dal/gen_plot.m`
