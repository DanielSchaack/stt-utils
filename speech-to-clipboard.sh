#!/bin/bash

file_path=$(dirname $(readlink -f $BASH_SOURCE))

cd "$file_path"
if [[ ! -d "./venv" ]]; then
    python -m venv ./venv
    source ./venv/bin/activate
    pip install -r ./requirements.txt
    deactivate
fi

source ./venv/bin/activate
python3 -u ./speech-to-clipboard.py
deactivate
cd - >/dev/null

