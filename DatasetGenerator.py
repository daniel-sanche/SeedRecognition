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
    configPath:     a path to the slice_config file which assiciated folders with classes
    validClasses:   the seed classes we are interested in. Others will be ignored

Returns:
    0:  dictionary with classes as keys, and lists of folders as values
"""
def getFolderLabels(configPath, validClasses=[25, 4, 23, 12, 22, 1, 6, 21, 24, 3, 15, 11, 2, 10, 5]):
    labelDict = {}
    thisLabel = 0
    with open(configPath) as f:
        for line in f:
            number, foldername = line.rstrip().split()
            if number == '1':
                thisLabel += 1
            if thisLabel in validClasses:
                labelDict[foldername] = thisLabel
    return labelDict


"""
Creates a set of folders to hold a generated dataset
The source dataset is divided into bins, so that each bin has a different set of source files to prevent testing
and training on the same images. Each bin will be initialized with a source_images.csv file, containing information
about the files assigned to this bin. Each bin should have roughly equivalent class distributions

Params:
    dataDir:    the directory that holds the dataset
    dirBaseName:    the base name given to the directory. A bin number will be appended to the end
    configPath: a path to the slice_config file which assiciated folders with classes
    numBins:    the number of bins to divide the resulting images into
                used for training and validation sets. Each bin will try to have an equal number of all classes
"""
def createFileBins(dataDir, dirBaseName="./Training_Bin", configPath="./class_map.txt", num_Bins=1):
    #check if directories exist
    if os.path.exists(dirBaseName) or os.path.exists(dirBaseName+"1"):
        print("bins named " + dirBaseName + " already exist")
        return
    #otherwise, create them
    fileList = []
    for i in range(num_Bins):
        if num_Bins > 1:
            newDirName = dirBaseName+str(i+1)
        else:
            newDirName = dirBaseName
        os.mkdir(newDirName)
        fileName = os.path.join(newDirName, "source_images.csv")
        fileList.append(open(fileName, 'w'))

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
    numNotEmpty = 1
    i = 0
    while numNotEmpty > 0:
        numNotEmpty = 0
        outFile = fileList[i%num_Bins]
        for key in classDict:
            thisList = classDict[key]
            if len(thisList) > 0:
                numNotEmpty += 1
                outFile.write(thisList.pop() + "\t" + str(key) + "\n")
        i += 1


"""
a python generator function that reads image paths in order from the file list,
and reads the images into memory. Also performs image scaling and conversion to
a float representation. Will loop through files indefinitely

Params:
    dataDir:    the directory that holds the raw dataset
    binDir:     the path to the bin we want to load images for. Created with "createFileBins"
    imageSize:  a vector of 3 values that represents the size all images should be scaled to
                ex, [500, 500, 3]
    asFloat:    determines whether images should be converted to a float representation

Yields:
    0:  the image loaded from disk
    1:  the image's class number
    2:  the image's relative file path in dataset folder
"""
def rawImageLoader(dataDir, binDir, imageSize=[224, 224, 3], asFloat=True):
    sourceFilePath=os.path.join(binDir,"source_images.csv")
    if not os.path.exists(sourceFilePath):
        print("Could not find file %s" % sourceFilePath)
        return
    while True:
        with open(sourceFilePath) as file:
            for line in file:
                fileSuffix, classNum = line.rstrip().split()
                fileName = os.path.join(dataDir, fileSuffix)
                img = imread(fileName)
                img = imresize(img, imageSize)
                if asFloat:
                    img = img / 255.0
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
        augmentedImg, metadata = ModifyImage(img, classNum, seed=seed)
        seed = seed + 1
        metadata["class"] = classNum
        metadata["origImgPath"] = filePath
        yield augmentedImg, metadata

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
            if i % 100 == 0:
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


#this function will generate file lists, then load and augment images from the file lists, and save them to new folders
#file lists contain the path to an image, and it's class name
if __name__ == "__main__":
    dataset_path = "/home/sanche/Datasets/Seed_Raw"
    dirBaseName="Generated_Bin"
    numBins=2
    numImagesPerBin = 10000
    imageSize = [224, 224, 3]

    createFileBins(dataset_path, dirBaseName=dirBaseName, num_Bins=numBins)
    for i in range(1, numBins+1):
        binDir = dirBaseName+str(i)
        imgLoader = rawImageLoader(dataset_path, binDir, imageSize=imageSize)
        augmentor = imageAugmentor(imgLoader)
        print(datetime.now().time())
        generatedImageSaver(augmentor, numImages=numImagesPerBin, imageDir=binDir)
        print(datetime.now().time())
