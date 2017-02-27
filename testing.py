import DataLoader
from vgg16 import vgg16
import os
import numpy as np

if __name__ == "__main__":
    imageRoot = "/home/sanche/Datasets/Seed_Test"

    vgg = vgg16('vgg16_weights.npz')
    batchSize = 45
    saveInterval = float("inf")

    filesCount = 0
    successCount = 0.0
    for dirNum in range(1,7):
        thisDir = "p" + str(dirNum) + "_45_first"
        imageDir = os.path.join(imageRoot, thisDir)
        fileReader = DataLoader.groundTruthReader(os.path.join(imageDir, 'p'+str(dirNum)+'_groundtruth.txt'))
        for imageBatch, classBatch in DataLoader.batchLoader(imageDir, fileReader,batchSize=batchSize):
            predictions, success = vgg.test(imageBatch, classBatch.reshape([batchSize,]))
            result = (classBatch.reshape([batchSize]), predictions, success)
            print("truth: %s, prediction: %s successRate: %f" % result)
            filesCount += batchSize
            successCount += (batchSize * success)
    print ("total success: %f in %d files" % (successCount/filesCount, filesCount))
