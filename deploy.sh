#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

rm -rf build/
rm -rf dist/
python3 setup.py sdist
twine upload dist/*
