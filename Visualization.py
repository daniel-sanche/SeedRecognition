import numpy as np
from scipy.misc import imresize, imsave


"""
combines a portion of the imageMat ino an image file that can be displayed

Params:
    imageMat:   the numpy array of images we want to display
    numRows:    the number of rows to display
    numCols:    the number of columns to display
    fileName:   the name to save the output file with
    maxImgSize: the size of each side of each block in the image grid
"""
def visualizeImages(imageMat, numRows=5, numCols=10, fileName="images.png", maxImgSize=64):
    imageSize = [imageMat.shape[1], imageMat.shape[2], imageMat.shape[3]]
    CombinedImage = np.ones([imageSize[0]*numRows, imageSize[1]*numCols, imageSize[2]])
    rowStart = 0
    i=0
    for r in range(numRows):
        rowEnd = rowStart + imageSize[0]
        RowImage = np.zeros([imageSize[0], imageSize[1]*numCols, imageSize[2]])
        lastStart = 0
        for c in range(numCols):
            thisImage = imageMat[i,:,:,:]
            end = lastStart + imageSize[1]
            RowImage[:, lastStart:end, :] = thisImage
            lastStart = end
            i=i+1
        CombinedImage[rowStart:rowEnd,:,:] = RowImage
        rowStart = rowEnd
    if maxImgSize is not None:
        CombinedImage = imresize(CombinedImage, [maxImgSize*numRows, maxImgSize*numCols])
    imsave(fileName, CombinedImage)