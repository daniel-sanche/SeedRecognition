import os
import numpy as np
from scipy.misc import imread, imsave, imresize
from skimage import exposure
from scipy.ndimage import rotate, interpolation
import random

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
    centerX:    the x point of the center of the image, relative to the image frame
                -1 specifies the left edge, 1 specifies the right edge
    centerY:    the y point of the center of the image, relative to the image frame
                -1 specifies the top edge, 1 specifies the bottom edge

Returns:
    0:  a numpy array consisting of imageMat with a graidnet mask applied to simulate a light source
"""
def addLighting(imageMat, radPercent=1.1, centerX=0, centerY=0):
    largerSide = max(imageMat.shape[1], imageMat.shape[2])
    half = int(largerSide/2)
    xCenter = (half * centerX) + half
    yCenter = (half * centerY) + half
    grad = _makeGradMask(largerSide, radius=largerSide * radPercent, center=[xCenter, yCenter])
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
def addGausianNoise(imageMat, mean=0, std=0.05):
    noise = np.random.normal(mean, std, imageMat.shape)
    return imageMat + noise


"""
Adjusts the contrast values for a set of images

Params:
    imageMat:       the input numpy array of images
    meanIntensity:  the new mean intensity value
    spread:          the range of values on either side of the mean

Returns:
    0:  a numpy array consisting of imageMat scaled to the new intensity values
"""
def adjustContrast(imageMat, meanIntensity=0.5, spread=0.5):
    minIntensity = max(meanIntensity - spread, 0)
    maxIntensity = min(meanIntensity + spread, 1)
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

"""
Mirrors the image left/right or up/down

Params:
    imageMat:   the numpy array of images to mirror
    mirrorLR:   a bool indicating whether to mirror the image left/right
    mirrorUD:   a bool indicating whether to mirror the image up/down

Returns:
    0:  a numpy array consisting of imageMat mirrored in the requested directions
"""
def mirrorImage(imageMat, mirrorLR=True, mirrorUD=True):
    if mirrorUD:
        resultMat = imageMat[:,::-1,:]
    else:
        resultMat = np.array(imageMat)
    if mirrorLR:
        resultMat = resultMat[:,:,::-1]
    return resultMat

"""
Shinks the size of the seed in the image, to simulate phots taken at different distances
The returned images will be the same size, but the seed will be smaller
Accomplished by adding padding to the original image, and then resizing to the proper size

Params:
    imageMat:   the numpy array of images to shrink
    padPercent: the amount of padding to apply relative to the size of the original image
                should be a value > 0
                ex: 1 = the image will be shrunk to half it's size, because equal padding is added

Returns:
    0:  a numpy array consisting of imageMat with the seeds shrunk
"""
def shrinkSeed(imageMat, padPercent=1):
    if padPercent <= 0:
        return np.array(imageMat)
    #pad the seed image, then resize to make seed appear smaller
    padValWidth = int(round((imageMat.shape[1] * padPercent)/2))
    padValHeight = int(round((imageMat.shape[2] * padPercent) / 2))
    resultMat = np.lib.pad(imageMat, ((0,0), (padValWidth, padValWidth), (padValHeight, padValHeight), (0,0)), mode="edge")
    resultMat = interpolation.zoom(resultMat, [1, imageMat.shape[1]/resultMat.shape[1], imageMat.shape[2]/resultMat.shape[2], 1])
    return resultMat[:,:imageMat.shape[1],:imageMat.shape[2],:]

"""
Translates the image

Params:
    imageMat:   the numpy array of images to translate
    xDelta:     the new position for the x center of the image
                -1 specifies the left edge, 1 specifies the right edge
                the old center point will be moved to this position on the image window
    yDelta:     the new position for the y center of the image
                -1 specifies the bottom edge, 1 specifies the top edge
                the old center point will be moved to this position on the image window

Returns:
    0:    a numpy array consisting of imageMat with the seeds translated
"""
def translate(imageMat, xDelta=0.5, yDelta=0.5):
    #add padding to shift the center point
    xVal = int(round((imageMat.shape[1] * abs(xDelta))))
    yVal = int(round((imageMat.shape[2] * abs(yDelta))))
    if xDelta > 0:
        xPad = (xVal, 0)
    else:
        xPad = (0, xVal)
    if yDelta > 0:
        yPad = (0, yVal)
    else:
        yPad = (yVal, 0)
    resultMat = np.lib.pad(imageMat, ((0, 0), yPad, xPad, (0, 0)), mode="edge")
    xMin = int(round((resultMat.shape[1]/2) - (imageMat.shape[1]/2)))
    xMax = xMin + imageMat.shape[1]
    yMin = int(round((resultMat.shape[2]/2) - (imageMat.shape[2]/2)))
    yMax = yMin + imageMat.shape[2]
    return resultMat[:,xMin:xMax, yMin:yMax,:]


def generateImages(baseImages, seed=None,
                   numRotations=4, rotationRange=[0,1],
                   numContrast=5, contrastProb=0.3, contrastMeanRange=[0.2, 0.6], contrastSpreadRange=[0.3, 0.5],
                   rGammaProb=0.3, gGammaProb=0.5, bGammaProb=0.5,
                   shrinkProb=0.3, shrinkRange=[0.5, 1],
                   numDifferentTranslations=5, translateProb=0.5, translateXRange=[-1,1], translateYRange=[-1,1],
                   numLighting=0, lightingProb=0.2, lightingRadRange=[0.9, 1.3], lightingXRange=[-0.5,0.5],lightingYRange=[-0.5,0.5],
                   noiseProb=0.4, noiseMeanRange=[0.4, 0.6], noiseStdRange=[0.03,0.15]):
    if seed is None:
        seed = int(random.random() * 4000000000)
        print ("seed used: " + str(seed))
    random.seed(seed)
    np.random.seed(seed)

    #add mirrored versions to the base images
    baseImages = np.concatenate((baseImages, mirrorImage(baseImages, True, True),
                                 mirrorImage(baseImages, False, True), mirrorImage(baseImages, True, False)), axis=0)
    #assign random rotations to the base images
    imageSet = [baseImages]
    for i in range(numRotations):
        rotations = rotateImage(baseImages, random.uniform(rotationRange[0], rotationRange[1]))
        imageSet += [rotations]
    baseImages = np.concatenate(imageSet)
    #randomize
    np.random.shuffle(baseImages)

    #adjust contrast in subset of images
    if numContrast > 0:
        subset = imageMat[:int(imageMat.shape[0]*contrastProb),:,:,:]
        batches = np.array_split(subset, numContrast)
        imageSet = [baseImages]
        for batch in batches:
            result = adjustContrast(batch, meanIntensity=random.uniform(contrastMeanRange[0], contrastMeanRange[1]),
                                        spread=random.uniform(contrastSpreadRange[0], contrastSpreadRange[1]))
            imageSet += [result]
        baseImages = np.concatenate(imageSet)
        #shuffle so contrast images are mixed in with others
        np.random.shuffle(baseImages)

    #adjust color channels in subset of images
    subset = imageMat[:int(imageMat.shape[0]*rGammaProb),:,:,:]
    result = gammaColorChannels(subset, R=True, G=False, B=False)
    baseImages = np.concatenate((baseImages, result))
    np.random.shuffle(baseImages)
    subset = imageMat[:int(imageMat.shape[0]*gGammaProb),:,:,:]
    result = gammaColorChannels(subset, R=False, G=True, B=False)
    baseImages = np.concatenate((baseImages, result))
    np.random.shuffle(baseImages)
    subset = imageMat[:int(imageMat.shape[0]*bGammaProb),:,:,:]
    result = gammaColorChannels(subset, R=False, G=False, B=True)
    baseImages = np.concatenate((baseImages, result))
    np.random.shuffle(baseImages)

    #adjust scale in subset of images
    subset = imageMat[:int(imageMat.shape[0] * shrinkProb), :, :, :]
    result = shrinkSeed(subset, random.uniform(shrinkRange[0], shrinkRange[1]))
    baseImages = np.concatenate((baseImages, result))
    np.random.shuffle(baseImages)

    #translate in subset of images
    if numDifferentTranslations > 0:
        subset = imageMat[:int(imageMat.shape[0] * translateProb), :, :, :]
        batches = np.array_split(subset, numDifferentTranslations)
        imageSet = [baseImages]
        for batch in batches:
            result = translate(batch, xDelta=random.uniform(translateXRange[0], translateXRange[1]),
                               yDelta=random.uniform(translateYRange[0], translateYRange[1]))
            imageSet += [result]
        baseImages = np.concatenate(imageSet)
        np.random.shuffle(baseImages)

    #add lighting to a subset of images
    if numLighting > 0:
        subset = imageMat[:int(imageMat.shape[0] * lightingProb), :, :, :]
        batches = np.array_split(subset, numLighting)
        imageSet = [baseImages]
        for batch in batches:
            result = addLighting(batch, radPercent=random.uniform(lightingRadRange[0], lightingRadRange[1]),
                                 centerX=random.uniform(lightingXRange[0], lightingXRange[1]),
                                 centerY=random.uniform(lightingYRange[0], lightingYRange[1]))
            imageSet += [result]
        baseImages = np.concatenate(imageSet)
        np.random.shuffle(baseImages)

    #add noise to subset of images
    subset = imageMat[:int(imageMat.shape[0] * noiseProb), :, :, :]
    result = addGausianNoise(subset, mean=random.uniform(noiseMeanRange[0], noiseMeanRange[1]),
                             std=random.uniform(noiseStdRange[0], noiseStdRange[1]))
    baseImages = np.concatenate((baseImages, result))
    np.random.shuffle(baseImages)

    return baseImages


if __name__ == "__main__":
    imageDir = "/Users/Sanche/Datasets/Seeds_Xin"
    imageMat = getImagesFromDir(imageDir, imageSize=[64, 64, 3])
    #imageMat = generateImages(imageMat, seed=923579038)
    imageMat = generateImages(imageMat)
    print(imageMat.shape)
    visualizeImages(imageMat, fileName="generated.png", numRows=20, numCols=20)




