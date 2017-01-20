import pickle
import os
import random
from scipy.misc import imread, imresize
import pandas as pd
from DataAugmentation import  ModifyImage
from scipy.misc import imsave
from datetime import  datetime

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
def rawImageLoader(dataDir, configPath, imageSize=[224, 224, 3], asFloat=True, fileListPath="./fileList.csv"):
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
a python generator function that yields augmented images.
Uses rawImageLoader, but preforms various augmentations on the loaded images.
Will load unique files indefinitely

Params:
    rawImageLoader: an instance of a rawImageLoader, used to load images from disk to augment
    seed:           if set, the same image sequence will be generated each run

Yields:
    0:  a numpy matrix containing an image
    1:  a dict containting metadata about the image
"""
def imageAugmentor(rawImageLoader, seed=None):
    if seed is None:
        seed = int(random.random() * 4000000000)
        print("seed used: " + str(seed))
    while True:
        img, classNum, filePath = next(rawImageLoader)
        augmentedImg, metadata = ModifyImage(img, seed=seed)
        seed = seed + 1
        metadata["class"] = classNum
        metadata["origImgPath"] = filePath
        yield img, metadata

"""
This function will create the generated image dataset
It pulls augmented images out of an imageAugmentor generator, and saves them to disk, along with relevant metadata
Will also save an index, so loading the images later will be easy
Can be called multiple times, and it will add new images without touching the existing ones

Params:
    imageGenerator: an instance of imageAugmentor to supply us with augmented images
    numImages:      the number of images to add to the directory
    imageDir:       the directory to save images in
    logFileName:    the name of the log file, containing metadata about each generated image
    indexFileName:  the name of the index file, which contains a list of filenames and their respective classes
                    this file is used to easily load images in order without searching through directories
"""
def generatedImageSaver(imageGenerator, numImages=100, imageDir="./GeneratedImages", logFileName="metadata.csv", indexFileName="index.tsv"):
    if not os.path.exists(imageDir):
        os.mkdir(imageDir)
    with open(os.path.join(imageDir, indexFileName), 'a') as index:
        logList = []
        for i in range(numImages):
            print("%d /%d" % (i, numImages))
            newImage, logs = next(imageGenerator)
            origFileName = logs["origImgPath"].replace("/", "|")
            seedUsed = str(logs["seedVal"])
            fileName = seedUsed + "|" + origFileName
            logs["fileName"] = fileName
            imsave(os.path.join(imageDir, fileName), newImage)
            logList.append(logs)
            index.write(logs["class"] + "\t" + fileName + "\n")
        logDf = pd.DataFrame(logList)
        logFilePath = os.path.join(imageDir, logFileName)
        logDf.to_csv(logFilePath, mode='a', header=(not os.path.exists(logFilePath)), index=False)



if __name__ == "__main__":
    dataset_path = "/home/sanche/Datasets/Seed_Images"
    config_path = dataset_path + "/slice_config"

    rawImageLoader = rawImageLoader(dataset_path, config_path, [224, 224, 3])
    augmentor = imageAugmentor(rawImageLoader)
    print(datetime.now().time())
    generatedImageSaver(augmentor, numImages=300)
    print(datetime.now().time())