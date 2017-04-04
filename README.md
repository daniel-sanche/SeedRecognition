## About the Project

This project is an attempt to use the VGG-16 CNN architecture to classify a dataset of seed images. A large part of the project consisted of creating image augmentation scripts to expand our available dataset. The project is building off of work done by Xin Yi at the University of Saskatchewan

## Network Weights

Because of their large file size, VGG-16’s initial weights are not available in this repository. Instead, they can be obtained on the (public internet)[https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3]

## Python Scripts

- DataLoader.py
  - contains functions necessary to load image files from disk
  - includes helper functions to load batches of images in numpy arrays of a requested size
- DataAugmentation.py
  - contains functions to augment images in the source image dataset, and save them to a new directory
  - performs transformations such as rotation, translation, noise application, and lighting simulation
- DatasetGenerator.py
  - used to generate a large dataset of training images
  - loads source images from disk, runs them through random augmentation functions, and saves the result to a new file
  - also creates csv files as an index, and to log information about the images in the new dataset
- kerasVGG.py
  - the actual implementation of the neural network, including functionality to train, predict, and test on input datasets

## Results

Included in this repository is a file called "Project Paper.pdf”, which details the results of the project