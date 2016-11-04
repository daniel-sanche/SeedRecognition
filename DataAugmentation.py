import os
import numpy as np
from scipy.misc import  imread

def getImagesFromDir(imageDir, imageSize=[640, 700, 3]):
    pathList = []
    for root,dir, files in os.walk(imageDir):
        for file in files:
            if ".png" in file:
                pathList += [os.path.join(root,file)]
    matSize = [len((pathList))] + imageSize
    imageMat = np.zeros(matSize)
    i=0
    for path in pathList:
        img = imread(path)
        imageMat[i,:,:,:] = img
        i=i+1
    return imageMat


if __name__ == "__main__":
    imageDir = "/Users/Sanche/Datasets/Seeds_Xin"
    imageMat = getImagesFromDir(imageDir)

