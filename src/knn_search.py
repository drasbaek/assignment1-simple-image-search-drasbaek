""" knn_search.py
Author:
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script forms a more advanced image search that extracts features from images using VGG16 and then uses a k-nearest-neighbors algorithm to find the most similar images.
    It is possible to specify the filename for the image as well as the number of similar images to return.

Usage:
   $ python src/knn_search.py
"""

# import packages
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from utils import *


def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16) (Taken directly from Notebook used in class, session 10)

    Args:
        img_path (str): path to image
        model: pretrained model

    Returns:
        normalized_features (numpy.ndarray): normalized feature representation of image

    """
    # define input image shape 
    input_shape = (224, 224, 3)

    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    
    # preprocess image
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    
    # flatten
    flattened_features = features.flatten()
    
    # normalise features
    normalized_features = flattened_features / norm(features)
    
    return normalized_features

def knn_most_similar_df(args, indices, distances, all_files, outpath):
    """
    Saves a dataframe with the most similar images to the main image.

    Args:
        args (argparse.ArgumentParser): arguments from argparse
        indices (numpy.ndarray): indices of the most similar images
        distances (numpy.ndarray): distances of the most similar images
        all_files (list): list of the similar images
        outpath (pathlib.PosixPath): path to output folder
    """

    # squeeze the arrays
    indices = indices.squeeze().tolist()
    distances = distances.squeeze().tolist()

    # get the filenames of the nearest neighbours
    filenames = [all_files[i].split(os.path.sep)[-1] for i in indices]

    # create dataframe
    df = pd.DataFrame({"filename": filenames, 
                       "distance": distances})
    
    # add main image as first row
    df.loc[0] = [f"{args.filename} (target)", 0]
    
    # save the dataframe as csv
    df.to_csv(outpath / "knn" / "knn_most_similar.csv", index=False)


def knn_plot_most_similar(inpath, args, indices, distances, all_files, outpath):
    """
    Plots the most similar images to the main image.
    The main image is marked with a red border.

    Args:
        inpath (pathlib.PosixPath): path to input folder
        args (argparse.ArgumentParser): arguments from argparse
        indices (numpy.ndarray): indices of the most similar images
        distances (numpy.ndarray): distances of the most similar images
        all_files (list): list of the similar images
        outpath (pathlib.PosixPath): path to output folder
    """

    # define main image name
    main_image_name = args.filename

    # squeeze the arrays and remove first element (the main image)
    indices = indices.squeeze().tolist()[1:]
    distances = distances.squeeze().tolist()[1:]

    # round distances to look nice in plot
    distances = [round(distance, 2) for distance in distances]

    # get the filenames of the nearest neighbours
    filenames = [all_files[i].split(os.path.sep)[-1] for i in indices]

    # get the main image
    main_image = cv2.imread(os.path.join(inpath, args.filename))

    # convert colors on main image
    main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

    # create figure
    fig = plt.figure(figsize=(12, 12))

    # get number of images
    n_images = args.top_n + 1

    # set subplot based on number of images
    ax = fig.add_subplot(n_images//3+1, 3, 1)

    # add the main image
    ax.imshow(main_image)

    # add title to original image including its name
    ax.set_title(f"Chosen image: {main_image_name}", color = "red")
    ax.axis("off")

    # add a mask on the edges that has same shape as image
    ax.add_patch(plt.Rectangle((0, 0), main_image.shape[1], main_image.shape[0], fill=False, edgecolor='red', lw=5))

    # add the most similar images
    for i, distance in enumerate(distances):
        ax = fig.add_subplot(n_images//3+1, 3, i+2)
        
        # get path to load image
        image_path = os.path.join(inpath, filenames[i])
        
        # load the image
        image = cv2.imread(image_path)

        # convert colors
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # plot the image
        ax.imshow(image)

        # add title to image including its name and distance
        ax.set_title(f"{filenames[i]} Similarity: {distance}")

        # remove axis
        ax.axis("off")

    # save the figure
    fig.savefig(outpath / "knn" / "knn_most_similar.png", bbox_inches='tight')


def main():
    # get paths
    inpath, outpath = define_paths()

    # run input parse
    args = input_parse()

    # check if inputs are okay
    input_checker(inpath, args)

    # get main image path with inpath
    main_file = os.path.join(inpath, args.filename)

    # initialize VGG16
    model = VGG16(weights='imagenet', 
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3))

    # extract features from the image
    main_features = extract_features(main_file, model)

    # get all image paths
    all_files = get_image_paths(inpath)

    # extract features from all images
    feature_list = []
    for i in tqdm(range(len(all_files))):
        feature_list.append(extract_features(all_files[i], model))
    
    # fit the nearest neighbour model
    neighbors = NearestNeighbors(n_neighbors=args.top_n+1, 
                             algorithm='brute',
                             metric='cosine').fit(feature_list)
    
    # get the nearest neighbours for the main image
    distances, indices = neighbors.kneighbors([main_features])
    
    # get the most similar images
    knn_most_similar_df(args, indices, distances, all_files, outpath)

    # plot the most similar images
    knn_plot_most_similar(inpath, args, indices, distances, all_files, outpath)

if __name__ == "__main__":
    main()