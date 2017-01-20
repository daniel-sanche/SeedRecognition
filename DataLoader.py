import os
import random
import numpy as np
from scipy.misc import imread


"""
loads file and class information from the index file in the generated image directory
when the index file is finished, it will be shuffled and reading will continue

Params:
    indexFilePath:  the path of the index file for the generated images

Yeilds:
    [0]:    the name of the next file to read
    [1]:    the class label for the next file to read
"""
def indexReader(indexFilePath):
    allLines = [line.rstrip().split() for line in open(indexFilePath, 'r')]
    while True:
        random.shuffle(allLines)
        for classVal, fileName in allLines:
            yield fileName, classVal

"""
loads batches of images from the generated images directory

Params:
    imageDir:       the directory of the generated images
    indexFileName:  the name of the index file in the imageDir
    batchSize:      the number of images in each batch
    imageSize:      the size of the images being extracted
Yields:
    [0]:    a numpy matrix containing a batch of images
    [1]:    a numpy array of label values for each image
"""
def batchLoader(imageDir, indexFileName="index.tsv", batchSize=65):
    index = indexReader(os.path.join(imageDir, indexFileName))
    #find the file size
    imageSize =  imread(os.path.join(imageDir, next(index)[0])).shape
    while True:
        thisBatch = np.zeros([batchSize, imageSize[0], imageSize[1], imageSize[2]])
        labelBatch = np.zeros([batchSize, 1])
        for i in range(batchSize):
            fileName, classVal = next(index)
            thisImage = imread(os.path.join(imageDir, fileName))
            thisBatch[i,:,:,:] = thisImage
            labelBatch[i] = classVal
        yield thisBatch, labelBatch


if __name__ == "__main__":
    imageDir = "./GeneratedImages"
    for imageBatch, classBatch in batchLoader(imageDir):
        print (imageBatch.shape, classBatch.shape)