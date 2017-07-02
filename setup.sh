#!/usr/bin/env bash

# Get directory containing bash script
WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

VENV_HOME=$WORKSPACE/venv

# Delete previously built virtualenv
if [ -d $VENV_HOME ]; then
	echo "Removed existing virtualenv"
    rm -rf $VENV_HOME
fi

# Create virtualenv
echo "Setting up virtualenv"
virtualenv $VENV_HOME
. $VENV_HOME/bin/activate

# Install required packages
echo "Ready to install packages"
echo "Installing numpy"; pip install -q numpy
echo "Installing matplotlib"; pip install -q matplotlib
echo "Installing scipy"; pip install -q scipy
echo "Installing UnbalancedDataset"
pip install -q git+https://github.com/fmfn/UnbalancedDataset

echo "Installing PIL"
wget -qO- http://effbot.org/downloads/Imaging-1.1.7.tar.gz | tar xz -C $WORKSPACE
python $WORKSPACE/Imaging-1.1.7/setup.py install >/dev/null 2>&1
rm -r $WORKSPACE/Imaging-1.1.7

echo "Installing python-weka-wrapper javabridge"
pip install -q python-weka-wrapper
echo "Installing pandas"; pip install -q pandas
echo "Installing liac-arff"; pip install -q liac-arff

# Install packages
if [ -f $WORKSPACE/requirements.txt ]; then
	echo "Installing required packages"
	pip install -r requirements.txt -q
fi

# TODO: This is no longer required
# Install node
echo "Installing node"
wget -qO- http://nodejs.org/dist/node-latest.tar.gz | tar xz -C $WORKSPACE
cd $WORKSPACE/node-v*
./configure --prefix=$VENV_HOME >/dev/null 2>&1
make install >/dev/null 2>&1
rm -r $WORKSPACE/node-v*
deactivate
. $VENV_HOME/bin/activate

# TODO: This is no longer required
# Install materialize
echo "Installing materialize"
npm install -g materialize-css --silent
