#!/usr/bin/env bash
# create virtual environment called image_search_env
python3 -m venv image_search_env

# activate virtual environment
source ./image_search_env/bin/activate

# install requirements
echo "[INFO] Installing requirements..."
python3 -m pip install -r requirements.txt

# run search using color channels (default params)
echo "[INFO] Running color channel search..."
python3 src/color_search.py --filename "image_0333.jpg" --top_n 5

# run search using knn (default params)
echo "[INFO] Running knn search..."
python3 src/knn_search.py --filename "image_0333.jpg" --top_n 5

# deactivate virtual environment
deactivate