import pickle
import os
import random
from scipy.misc import imread, imresize
import numpy as np
from DataAugmentation import  ModifyImage
from Visualization import visualizeImages
import pandas as pd
"""
creates a label dict file that holds the names of each folder associated with each class
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

def augmentedImageGenerator(dataDir, configPath, batchSize=10, imageSize=[200,200,3], seed=None):
    imageLoader = rawImageLoader(dataDir, configPath, imageSize)
    if seed is None:
        seed = int(random.random() * 4000000000)
        print("seed used: " + str(seed))
    while True:
        loglist = []
        batchMat = np.zeros([batchSize]+imageSize)
        for i in range(batchSize):
            img, classNum, filePath = next(imageLoader)
            augmentedImg, logs = ModifyImage(img, seed=seed)
            batchMat[i, :, :, :] = augmentedImg
            logs["class"] = classNum
            logs["path"] = filePath
            loglist += [logs]
            seed = seed + 1
        yield batchMat, pd.DataFrame(loglist)




if __name__ == "__main__":
    dataset_path = "/Users/Sanche/Datasets/Seeds_Full"
    config_path = dataset_path + "/slice_config"

    for imgMat, logDf in augmentedImageGenerator(dataset_path, config_path, batchSize=16):
        logDf.to_csv("logs.csv")
        visualizeImages(imgMat, fileName="generated.png", numCols=4, numRows=4)
        break