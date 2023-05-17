""" utils.py
Author:
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script contains utility functions for the image search engine.
    These are the shared functions that are used in both the color search and the knn search.
"""

# import packages
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import cv2
import numpy as np
import argparse
from pathlib import Path


def define_paths():
    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "data" / "flowers"

    # define output dir
    outpath = path.parents[1] / "out"

    return inpath, outpath


def input_parse():
    """
    Parse command line arguments to script.
    It is possible to specify the filename for the image as well as the number of similar images to return.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    # create a parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("-f", "--filename", type=str, default = "image_0333.jpg", help="Name of the image to compare to.")
    parser.add_argument("-n", "--top_n", type=int, default = 5, help="Number of images to return.")

    # parse the arguments
    args = parser.parse_args()

    # return the arguments
    return args


def input_checker(inpath, args):
    """
    Check if the input arguments are correct and return error message if not.
    
    Args:
        inpath (str): Path to the input directory.
        args (argparse.Namespace): Parsed arguments.
    """
    # check if the top_n argument is a positive integer that also does not exceed the number of images in the inpath directory
    if args.top_n <= 0 or args.top_n > len(os.listdir(inpath)):
        print("The top_n argument is not a positive integer or it exceeds the number of images in the inpath directory. Check the top_n argument again.")
        sys.exit()

    # check if the filename is in the inpath directory
    if args.filename not in os.listdir(inpath):
        print("The filename is not in the inpath directory. Check the filename and filetype again.")
        sys.exit()


def get_image_paths(data_dir: str):
    """
    Get the paths to the images in the data directory.

    Args:
        data_dir (str): Path to the data directory.
    
    Returns:
        image_paths (list): List of image paths.
    """
    images = os.listdir(data_dir)

    # add the full path to the image name
    image_paths = [os.path.join(data_dir, image) for image in images]
    
    # return the image paths
    return image_paths
    