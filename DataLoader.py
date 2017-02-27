import os
import random
import numpy as np
from scipy.misc import imread, imresize


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

def groundTruthReader(filePath):
    allLines = [line.rstrip().split() for line in open(filePath, 'r')]
    for thisLine in allLines:
        fileName = thisLine[0].zfill(6) + ".png"
        yield fileName, thisLine[2]

"""
loads batches of images from the generated images directory

Params:
    imageDir:       the directory of the generated images
    indexReader:  an index generator, that yeilds file names and class values
    batchSize:      the number of images in each batch
    imageSize:      the size of the images being extracted
Yields:
    [0]:    a numpy matrix containing a batch of images
    [1]:    a numpy array of label values for each image
"""
def batchLoader(imageDir, indexReader, imageSize=224, batchSize=64):
    #find the file size
    while True:
        thisBatch = np.zeros([batchSize, imageSize, imageSize, 3])
        labelBatch = np.zeros([batchSize, 1], dtype=int)
        for i in range(batchSize):
            fileName, classVal = next(indexReader)
            thisImage = imread(os.path.join(imageDir, fileName))
            if thisImage.shape[0] != imageSize or thisImage.shape[1] != imageSize:
                thisImage = imresize(thisImage, (imageSize, imageSize))
            thisBatch[i,:,:,:] = thisImage
            labelBatch[i] = classVal
        yield thisBatch, labelBatch


if __name__ == "__main__":
    imageDir = "./GeneratedImages"
    for imageBatch, classBatch in batchLoader(imageDir):
        print (imageBatch.shape, classBatch.shape)
