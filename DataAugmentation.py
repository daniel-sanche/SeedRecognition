import os
import numpy as np
from scipy.misc import imread, imsave, imresize
from skimage import exposure
from scipy.ndimage import rotate

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

"""
loads a set of images from a directory. Will find all .png files in any nested directory
Images will be stored as floats between 0 and 1

Params:
    imageDir:   the root directory to find images in
    imageSize:  images will be resized to this size
Returns:
    0:  a numpy array of images, stacked on the first dimension
"""
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
        img = imresize(img, imageSize)
        imageMat[i,:,:,:] = img
        i=i+1
    return imageMat / 255

"""
appplies a gamma transformation to color channels

Params:
    imageMat:   the numpy array of images to adjust
    R:          indicates whether to adjust the red channel
    G:          indicates whether to adjust the green channel
    B:          indicates whether to adjust the blue channel

Returns:
    0:  a numpy array of the new adjusted images
"""
def gammaColorChannels(imageMat, R=True, G=True, B=True):
    resultMat = np.array(imageMat)
    if R:
        resultMat[:,:,:,0] = np.power(imageMat[:,:,:,0],(2.2))
    if G:
        resultMat[:, :, :, 1] = np.power(imageMat[:, :, :, 1], (2.2))
    if B:
        resultMat[:, :, :, 2] = np.power(imageMat[:, :, :, 2], (2.2))
    return resultMat


"""
helper function to make a radial gradient mask with the proper attributes

Params:
    size:   the size of the sides of the square mask
    radius: the size of the radius of the radial gradient
    center: the center point of the gradient

Returns:
    0:  a size*size grayscale gradient image, that can be treated as a mask
"""
def _makeGradMask(size=100, radius=50, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)

"""
Adds simulated lighting in the form of a radial gradient mask to a set of images

Params:
    imageMat:   the numpy array of images to add lighting to
    radPercent: the size of the radius of the light, in relation to the size of the image
                larger radi are preferred to avoid excessive vignette effect
    center:     the center point of the simulated light on the image

Returns:
    0:  a numpy array consisting of imageMat with a graidnet mask applied to simulate a light source
"""
def addLighting(imageMat, radPercent=1.1, center=None):
    largerSide = max(imageMat.shape[1], imageMat.shape[2])
    grad = _makeGradMask(largerSide, radius=largerSide * radPercent, center=center)
    grad = imresize(grad, [imageMat.shape[1], imageMat.shape[2]])
    grad = np.tile(grad, (3,1,1)).transpose(1,2,0)
    return imageMat * grad

"""
Adds gaussian noise to a set of images

Params:
    imageMat:   the numpy array of images to add noise to
    mean:       the mean of the gaussian noise distribution
    std:        the standard deviation of the gaussian noise

Returns:
    0:  a numpy array consisting of imageMat with gaussian noise added
"""
def addGausianNoise(imageMat, mean=0, std=0.1):
    noise = np.random.normal(mean, std, [imageMat.shape[1], imageMat.shape[2], imageMat.shape[3]])
    return imageMat + noise


"""
Adjusts the contrast values for a set of images

Params:
    imageMat:       the input numpy array of images
    meanIntensity:  the new mean intensity value
    range:          the range of values on either side of the mean

Returns:
    0:  a numpy array consisting of imageMat scaled to the new intensity values
"""
def adjustContrast(imageMat, meanIntensity=0.5, range=0.5):
    minIntensity = max(meanIntensity - range, 0)
    maxIntensity = min(meanIntensity + range, 1)
    return exposure.rescale_intensity(imageMat, (minIntensity, maxIntensity))

"""
Rotates the set of input images
Fills in blank region with the nearest neighbour pixels, to retain consistent background

Params:
    imageMat:   the set of images to rotate
    rotationPercent:    the percent we want to rotate the image.
                        Will be multiplied by 360 to determine the number of degrees

Returns:
    0:  a numpy array consisting of imageMat rotated by rotationPercent*360 degrees
"""
def rotateImage(imageMat, rotationPercent=0.5):
    return rotate(imageMat, rotationPercent*360, axes=[1,2], reshape=False, mode="nearest")

if __name__ == "__main__":
    imageDir = "/Users/Sanche/Datasets/Seeds_Xin"
    imageMat = getImagesFromDir(imageDir, imageSize=[100, 100, 3])
    #visualizeImages(imageMat, fileName="orig.png")
    #gammaColorChannels(imageMat)
    imageMat = rotateImage(imageMat, 0.1)
    visualizeImages(imageMat, fileName="rotated.png")




