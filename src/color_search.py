""" color_search.py
Author:
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script forms a simple image search engine that compares the color histograms of images in a directory to a target image.
    It is possible to specify the filename for the image as well as the number of similar images to return.

Usage:
   $ python src/color_search.py
"""

# import packages
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import *

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to a single image.
    
    Returns:
        image (ndarray): Loaded image.
    """
    # load the image
    image = cv2.imread(image_path)

    # identify the image name
    image_name = image_path.split(os.path.sep)[-1]
    
    # return the image
    return image, image_name


def create_norm_hist(image):
    """
    Create a normalized histogram for the image.

    Args:
        image (ndarray): Input image.
    
    Returns:
        hist (ndarray): Normalized color histogram.
    """

    # extract histograms of channels
    hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    
    # do min-max normalization
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    # return the histogram
    return hist


def compare_hist(main_hist, hist):
    """
    Compare the main histogram to the histogram of the image.
    
    Args:
        main_hist (ndarray): Histogram of main image.
        hist (ndarray): Histogram of image to compare to
    
    Returns:
        score (float): Score of the Chi-Square comparison.
    """

    # compute the chi-squared distance
    score = cv2.compareHist(main_hist, hist, cv2.HISTCMP_CHISQR)

    # round score
    score = round(score, 2)
    
    # return the score
    return score


def find_most_similar(images_and_scores, top_n = 5):
    """
    Find the most similar images.
    Args:
        all_images_with_scores (dict): Dictionary with all the images and their scores.
        top_n (int): Number of images to return.
    
    Returns:
        most_similar_images (dict): List of most similar images.
    """

    # sort the dictionary by the values (lowest first)
    sorted_dict = sorted(images_and_scores.items(), key=lambda x: x[1])

    # remove the first element as it is the main image
    sorted_dict.pop(0)
    
    # get the top n images 
    most_similar_images = dict(sorted_dict[:top_n])
    
    # return the most similar images
    return most_similar_images


def most_similar_df(most_similar_images, main_image_name, outpath):
    """
    Create and saves a dataframe with the most similar images.
    
    Args:
        most_similar_images (dict): Dictionary with the most similar images.
        main_image_name (str): File name for main image.
        outpath (pathlib.PosixPath): Path to save the dataframe.
    """

    # create a dataframe
    df = pd.DataFrame(columns=["Filename", "Distance"])
    
    # add main image as first row
    df.loc[0] = [f"{main_image_name} (target)", 0]

    # add the most similar images
    for i, (image_name, score) in enumerate(most_similar_images.items()):
        df.loc[i+1] = [image_name, score]
    
    # save the dataframe as csv using outpath
    df.to_csv((outpath / "color_channels" / "colorchannels_most_similar.csv"), index=False)


def most_similar_plot(inpath, main_image, main_image_name, most_similar_images, outpath):
    """
    Plot the main image and the most similar images.
    The main image is marked with a red border.

    Args:
        inpath (pathlib.PosixPath): Path to input images.
        main_image (ndarray): Main image.
        main_image_name (str): File name for main image.
        most_similar_images (dict): Dictionary with the most similar images.
        outpath (pathlib.PosixPath): Path to save the plot.
    """
    
    # create a figure
    fig = plt.figure(figsize=(12, 12))

    # number of images
    n_images = len(most_similar_images) + 1
    
    # set subplot based on number of images
    ax = fig.add_subplot(n_images//3+1, 3, 1)

    # convert colors on main image
    main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

    # add the main image
    ax.imshow(main_image)

    # add title to original image including its name
    ax.set_title(f"Chosen image: {main_image_name}", color = "red")
    ax.axis("off")

    # add a mask on the edges that has same shape as image
    ax.add_patch(plt.Rectangle((0, 0), main_image.shape[1], main_image.shape[0], fill=False, edgecolor='red', lw=5))

    
    # add the most similar images
    for i, (image_name, score) in enumerate(most_similar_images.items()):
        ax = fig.add_subplot(n_images//3+1, 3, i+2)
        
        # get path to load image
        image_path = os.path.join(inpath, image_name)
        
        # load the image
        image = cv2.imread(image_path)

        # convert colors
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # plot the image
        ax.imshow(image)

        ax.set_title(f"{image_name.split(os.path.sep)[-1]} Similarity: {score}")

        ax.axis("off")

    # save plot as png
    plt.savefig(outpath / "color_channels" / "colorchannels_most_similar.png",  bbox_inches='tight')
    

def main():
    # define paths
    inpath, outpath = define_paths()

    # run input parse
    args = input_parse()

    # check if inputs are okay
    input_checker(inpath, args)

    # get main image path with inpath
    main_file = os.path.join(inpath, args.filename)

    # load the main image
    main_image, main_image_name = load_image(main_file)

    # create histogram for the main image
    main_hist = create_norm_hist(main_image)

    # get the image paths for all other images
    image_paths = get_image_paths(inpath)

    # dictionary to store the images and their scores
    other_images_and_scores = {}

    # loop over image path with progress bar
    for image_path in tqdm(image_paths, desc="Comparing images"):
    
        # load the image
        image, image_name = load_image(image_path)
    
        # create the normalized histogram
        hist = create_norm_hist(image)
    
        # compare to the main histogram
        score = compare_hist(main_hist, hist)
    
        # add the image and its score to the dictionary
        other_images_and_scores[image_name] = score

    # find the most similar images
    most_similar_images = find_most_similar(other_images_and_scores, top_n=args.top_n)

    # save plot of most similar images
    most_similar_plot(inpath, main_image, main_image_name, most_similar_images, outpath)

    # save dataframe of most similar images
    most_similar_df(most_similar_images, main_image_name, outpath)


if __name__ == "__main__":
    main()
