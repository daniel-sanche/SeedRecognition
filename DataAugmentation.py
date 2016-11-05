import os
import numpy as np
from scipy.misc import imread, imsave, imresize

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
    return imageMat / 255

def gammaColorChannels(imageMat, R=True, G=True, B=True):
    if R:
        imageMat[:,:,:,0] = np.power(imageMat[:,:,:,0],(2.2))
    if G:
        imageMat[:, :, :, 1] = np.power(imageMat[:, :, :, 1], (2.2))
    if B:
        imageMat[:, :, :, 2] = np.power(imageMat[:, :, :, 2], (2.2))

def makeGaussian(size=100, radius=50, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)

def addLighting(imageMat, center=None):
    largerSide = max(imageMat.shape[1], imageMat.shape[2])
    grad = makeGaussian(largerSide, radius=largerSide*1.1, center=center)
    grad = imresize(grad, [imageMat.shape[1], imageMat.shape[2]])
    grad = np.tile(grad, (3,1,1)).transpose(1,2,0)
    imageMat[:,:,:,:] = imageMat * grad


if __name__ == "__main__":
    imageDir = "/Users/Sanche/Datasets/Seeds_Xin"
    imageMat = getImagesFromDir(imageDir)
    #visualizeImages(imageMat, fileName="orig.png")
    #gammaColorChannels(imageMat)
    addLighting(imageMat)
    visualizeImages(imageMat, fileName="gradient_seeds.png")




