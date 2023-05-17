# Assignment 1: Simple Image Search
## Description
This repository includes the solution by *Anton Drasbæk Schiønning (202008161)* to assignment 1 in the course "Visual Analytics" at Aarhus University. <br>

It is used to complete an image search on an image database, such as the [Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/), to identify the most similar images to a selected one. To achieve this, two search methods are investigated: Using color channels and k-nearest neighbors on image features extracted using *VGG16*.
<br/><br/>

## Repository Tree

```
├── README.md         
├── assign_desc.md    
├── data/             
│   └── flowers.zip         -----> flowers.zip file with pictures of all flowers (should be unzipped)
├── out/              
│   ├── color_channels   
│   │   ├── colorchannels_most_similar.csv  -----> table of the most similar images to the main images and their similarity scores
│   │   └── colorchannels_most_similar.png  -----> visualization of the most similar images
│   └── knn
├── requirements.txt   
├── setup.sh           
└── src/               
    ├── color_search.py     -----> script for search based on color channels
    ├── knn_search.py       -----> script for search based on nearest neighboors for VGG16 features
    └── utils.py            -----> functions that are used for both image searches

```
<br/><br/>
## Usage
This project only assumes that you have Python3 installed. The file `flowers.zip` in `data` should be unpacked and inserted into the data directory. The "flowers" folder that you get is added to .gitignore, so it will not be pushed. <br>

To run the full analysis, including an image search using both color channels and KNN, run the `run.sh` file from the root directory:
```
bash run.sh
```
This will complete the following steps:
* Create and activate a virtual environment
* Install requirements to that environment
* Run the image search using color channels for the default image
* Run the image search using KNN for the default image
* Deactivate the environment
<br>

## Modified Usage
If you want to investigate a different picture or compare with more/less images, you can run a modified analysis. <br>

### Setup
Apart from unzipping `flowers.zip`, you must run the setup file from the root directory to install requirements and initialize a virtual environment:
```
bash setup.sh
```
### Run Modified Analysis
The adaptations to running an analysis with modifications are available through using the two arguments:
```
- f- --filename (default: image_0333.png)
- n- --top_n (default: 5)
```

These can be used for both the analysis for color channels and for the KNN as such:
```
# run analysis for image_0023 with 8 most similar images using both searching techniques
python src/color_search --filename "image_0023.jpg" --top_n 8
python src/knn_search --filename "image_0023.jpg" --top_n 8
```
<br>

## Exemplary Results
### Color Channel Search
![alt text](https://github.com/AU-CDS/assignment1-simple-image-search-drasbaek/blob/main/out/color_channels/colorchannels_most_similar.png?raw=True)

### KNN/VGG16-feature search
![alt text](https://github.com/AU-CDS/assignment1-simple-image-search-drasbaek/blob/main/out/knn/knn_most_similar.png?raw=True)

## Discussion


