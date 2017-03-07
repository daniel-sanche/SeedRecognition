import os
import random
import numpy as np
from scipy.misc import imread, imresize
from keras.utils.np_utils import to_categorical


"""
loads file and class information from the index file in the generated image directory
generator will keep loading files in order infinitely

Params:
    folderPath:  the path of the generated images folder we want to load
                 the folder is assumed to have an index file called index.tsv

Yeilds:
    [0]:    the path of the next file to read
    [1]:    the class label for the next file to read
"""
def generatedDatasetParser(folderPath):
    print("loading data from %s" % folderPath)
    indexFilePath = os.path.join(folderPath, "index.tsv")
    while True:
        with open(indexFilePath, 'r') as f:
            for thisLine in f:
                classVal, fileName = thisLine.rstrip().split()
                filePath = os.path.join(folderPath, fileName)
                yield filePath, classVal

"""
loads file and class information from the index file in the segmented image directory
when all files have been loaded, generator stops

Params:
    rootDir:   the root path of the segmenteeed images we want to load
                the folder is assumed to be structured with a set of 7 inner folders,
                each containing an index called "px_groundtruth.txt" that has class information

Yeilds:
    [0]:    the path of the next file to read
    [1]:    the class label for the next file to read
"""
def segmentedDatasetParser(rootDir):
    print("loading data from %s" % rootDir)
    for dirNum in range(1, 7):
        dirName = "p" + str(dirNum) + "_45_first"
        dirPath = os.path.join(rootDir, dirName)
        indexName = 'p' + str(dirNum) + '_groundtruth.txt'
        indexPath = os.path.join(dirPath, indexName)
        with open(indexPath, 'r') as f:
            for thisLine in f:
                thisLine = thisLine.rstrip().split()
                fileName = thisLine[0].zfill(6) + ".png"
                className = thisLine[2]
                filePath = os.path.join(dirPath, fileName)
                if os.path.exists(filePath):
                    yield filePath, className

"""
loads file and class information from the testing image directory
when all files have been loaded, generator stops

Params:
    rootDir:   the root path of the segmenteeed images we want to load
                the folder is assumed to be structured with a set of 7 inner folders,
                each containing an index called "px_groundtruth.txt" that has class information

Yeilds:
    [0]:    the path of the next file to read
    [1]:    the class label for the next file to read
"""
def testDatasetParser(rootDir):
    print("loading data from %s" % rootDir)
    for dirNum in range(1, 7):
        dirName = "p" + str(dirNum) + "_45_first"
        dirPath = os.path.join(rootDir, dirName)
        indexName = 'p' + str(dirNum) + '_groundtruth.txt'
        indexPath = os.path.join(dirPath, indexName)
        with open(indexPath, 'r') as f:
            for thisLine in f:
                thisLine = thisLine.rstrip().split()
                className = thisLine[2]
                folderName = thisLine[0].zfill(6)
                folderPath = os.path.join(dirPath, folderName)
                if os.path.exists(folderPath):
                    for imgFile in os.listdir(folderPath):
                        if ".png" in imgFile:
                            imgPath = os.path.join(folderPath, imgFile)
                            yield imgPath, className

"""
loads batches of images from the generated images directory

Params:
    imageDir:       the directory of the generated images
    indexReader:    an index generator, that yeilds file names and class values
    batchSize:      the number of images in each batch
    imageSize:      the size of the images being extracted
Yields:
    [0]:    a numpy matrix containing a batch of images
    [1]:    a numpy array of label values for each image
"""
def batchLoader(indexReader, imageSize=224, batchSize=64):
    #find the file size
    while True:
        thisBatch = np.zeros([batchSize, imageSize, imageSize, 3])
        labelBatch = np.zeros([batchSize, 1], dtype=int)
        for i in range(batchSize):
            filePath, classVal = next(indexReader)
            thisImage = imread(filePath)
            if thisImage.shape[0] != imageSize or thisImage.shape[1] != imageSize:
                thisImage = imresize(thisImage, (imageSize, imageSize))
            thisBatch[i,:,:,:] = thisImage
            labelBatch[i] = classVal
        yield thisBatch, labelBatch

"""
Finds the number of results in a generator object

Params:
    generator:    the generator to query
Returns:
    [0]:    the number of results in the generator
"""
def countResultsInGenerator(generator):
    count = 0
    for _ in generator:
        count += 1
        if count > 1000000:
            return float("inf")
    return count

"""
takes in a batch loader generator, and changes the labels to be in the one-hot format

Params:
    batchLoader:    a batch loader generator that yeilds imageBatch, labelBath pairs
    numClasses:     the number of classes possible for the labels
Yields:
    [0]:    a numpy matrix containing a batch of images
    [1]:    a numpy array of one-hot labels for each image
"""
def oneHotWrapper(batchLoader, numClasses=30):
    for img, label in batchLoader:
        label = label.reshape([img.shape[0],])
        oneHotLabels = to_categorical(label, nb_classes=numClasses)
        yield img, oneHotLabels