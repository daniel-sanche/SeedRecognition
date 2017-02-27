import DataLoader
from vgg16 import vgg16
import os
import numpy as np

if __name__ == "__main__":
    imageDir = "/home/sanche/Datasets/Seed_Test/p4_45_first/"

    vgg = vgg16('vgg16_weights.npz')
    batchSize = 45
    saveInterval = float("inf")

    i = 0
    fileReader = DataLoader.groundTruthReader(os.path.join(imageDir, 'p4_groundtruth.txt'))
    for imageBatch, classBatch in DataLoader.batchLoader(imageDir, fileReader,batchSize=batchSize):
        predictions, successRate = vgg.test(imageBatch, classBatch.reshape([batchSize,]))
        result = (classBatch.reshape([batchSize]), predictions, successRate)
        print("truth: %s, prediction: %s successRate: %f" % result)
