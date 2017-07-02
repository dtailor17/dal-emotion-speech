#!/usr/bin/env bash

# Get directory containing bash script
WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Display IP address
ip_addr="$( ip addr | grep 'state UP' -A2 | tail -n1 | awk '{print $2}' | cut -f1  -d'/' )"
echo "IP address: $ip_addr"

# Get virtual environment python path
PYTHON_PATH=$WORKSPACE/venv/bin/python

# Run server
$PYTHON_PATH run.py
