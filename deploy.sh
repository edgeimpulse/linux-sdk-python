#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

if [ ! -d ".venv" ]; then
    if [ -z "$SB_API_KEY" ]; then
        echo "Missing SB_API_KEY, set to a StableBuild API Key (required to install pinned build dependencies)"
        exit 1
    fi

    python3.9 -m venv .venv
    source .venv/bin/activate

    pip3 install \
        -i https://$SB_API_KEY.pypimirror.stablebuild.com/2023-01-30/ \
        twine
else
    source .venv/bin/activate
fi

rm -rf build/
rm -rf dist/
.venv/bin/python3 setup.py sdist
.venv/bin/twine upload dist/*

deactivate
