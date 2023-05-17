#!/usr/bin/env bash
# create virtual environment called image_search_env
python3 -m venv image_search_env

# activate virtual environment
source ./image_search_env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt