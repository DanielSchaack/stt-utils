#!/bin/bash

cleanup() {
    deactivate
}

source ./venv/bin/activate
python3 -u voice.py
cleanup

