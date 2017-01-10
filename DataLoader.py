import pickle
import os
import random
from scipy.misc import imread, imresize
import numpy as np
from DataAugmentation import  ModifyImage
from Visualization import visualizeImages
import pandas as pd

"""
finds all image folders associated with each class

Params:
    configPath:   a path to the slice_config file which assiciated folders with classes

Returns:
    0:  dictionary with classes as keys, and lists of folders as values
"""
def getFolderLabels(configPath):
    labelDict = {}
    thisLabel = -1
    with open(configPath) as f:
        for line in f:
            number, foldername = line.rstrip().split()
            if number == '1':
                thisLabel += 1
            labelDict[foldername] = thisLabel
    return labelDict


"""
Creates the file list
This file is a simple csv, with each line consisting of an image path, and it's class
THis file can then be iterated through to load images into memory for the next batch
The lines in the file alternate between each class in order, to ensure that each batch will
have a similar distribution of all classes to prevent bias

Params:
    dataDir:    the directory that holds the dataset
    configPath: a path to the slice_config file which assiciated folders with classes
    outPath:    the output path for the csv file
"""
def createFileList(dataDir, configPath, outputPath):
    #find all the folders associated with each class
    labelDict = getFolderLabels(configPath)
    #find the set of all files associated with each class
    classDict = {}
    for root, dirs, files in os.walk(dataDir):
        lastFolder = os.path.basename(root)
        if lastFolder in labelDict:
            classNum = labelDict[lastFolder]
            for file in files:
                if ".png" in file and "bw" not in file:
                    oldList = classDict.get(classNum, [])
                    classDict[classNum] = oldList + [os.path.join(lastFolder, file)]
    #randomize each list
    for key in classDict:
        random.shuffle(classDict[key])
    #print a csv file that steps through random samples of each class equally
    with open(outputPath, 'w') as outFile:
        numNotEmpty = 1
        while numNotEmpty > 0:
            numNotEmpty = 0
            for key in classDict:
                thisList = classDict[key]
                if len(thisList) > 0:
                    numNotEmpty += 1
                    outFile.write(thisList.pop() + "\t" + str(key) + "\n")


"""
a python generator function that reads image paths in order from the file list,
and reads the images into memory. Also performs image scaling and conversion to
a float representation. Will loop through files indefinitely

Params:
    dataDir:    the directory that holds the dataset
    configPath: a path to the slice_config file which assiciated folders with classes
    imageSize:  a vector of 3 values that represents the size all images should be scaled to
                ex, [500, 500, 3]
    asFloat:    determines whether images should be converted to a float representation
    fileListPath:    the path of the  file list csv file

Yields:
    0:  the image loaded from disk
    1:  the image's class number
    2:  the image's relative file path in dataset folder
"""
def rawImageLoader(dataDir, configPath, imageSize, asFloat=True, fileListPath="./fileList.csv"):
    if not os.path.exists(fileListPath):
        createFileList(dataDir, configPath, fileListPath)
    while True:
        with open(fileListPath) as file:
            for line in file:
                fileSuffix, classNum = line.rstrip().split()
                fileName = os.path.join(dataDir, fileSuffix)
                img = imread(fileName)
                img = imresize(img, imageSize)
                if asFloat:
                    img = img / 255
                yield img, classNum, fileSuffix

"""
a python generator function that yields batches of augmented images.
Uses rawImageLoader, but preforms various augmentations on the loaded images.
Will load unique files indefinitely

Params:
    dataDir:    the directory that holds the dataset
    configPath: a path to the slice_config file which assiciated folders with classes
    batchSize:  the number of images to include in the batch
    imageSize:  a vector of 3 values that represents the size all images should be scaled to
                ex, [500, 500, 3]
    seed:       if set, the same image sequence will be generated each run
    keepLogs:   determines whether logs are saved. If true, a csv file called logs.csv will be saved
                in the project directory containing information about each image used in each batch,
                and all augmentations performed on the images

Yields:
    0:  a numpy matrix containing a batch of images. Will be size [batchSize, imageSize[0],imageSize[1],imageSize[2]]
    1:  a 1D numpy array containing the class label for each image in the batch. Size [batchSize, 1]
"""
def augmentedImageGenerator(dataDir, configPath, batchSize=10, imageSize=[200,200,3], seed=None, keepLogs=True):
    imageLoader = rawImageLoader(dataDir, configPath, imageSize)
    logFile = "./logs.csv"
    if seed is None:
        seed = int(random.random() * 4000000000)
        print("seed used: " + str(seed))
    while True:
        loglist = []
        labelMat = np.zeros([batchSize, 1])
        batchMat = np.zeros([batchSize]+imageSize)
        for i in range(batchSize):
            img, classNum, filePath = next(imageLoader)
            augmentedImg, logs = ModifyImage(img, seed=seed)
            batchMat[i, :, :, :] = augmentedImg
            labelMat[i] = classNum
            seed = seed + 1
            if keepLogs:
                logs["class"] = classNum
                logs["path"] = filePath
                loglist += [logs]
        #save log file to disk
        if keepLogs:
            logDf = pd.DataFrame(loglist)
            logDf.to_csv(logFile, mode='a', header=(not os.path.exists(logFile)), index=False)
        yield batchMat, labelMat




if __name__ == "__main__":
    dataset_path = "/Users/Sanche/Datasets/Seeds_Full"
    config_path = dataset_path + "/slice_config"

    #delete old logs
    if os.path.exists("./logs.csv"):
        os.remove("./logs.csv")

    for imgMat, labelMat in augmentedImageGenerator(dataset_path, config_path, batchSize=16):
        visualizeImages(imgMat, fileName="generated.png", numCols=4, numRows=4)
        print("done")