import DataLoader
from kerasVGG import VGG
import os
import numpy as np

if __name__ == "__main__":
    imageRoot = "/home/sanche/Datasets/Seed_Test"
    checkpointName = "./keras_checkpoint.h5"

    vgg = VGG()
    vgg.loadWeights(checkpointName, None)
    batchSize = 45

    filesCount = 0
    successCount = 0.0
    fileReader = DataLoader.segmentedDatasetParser(imageRoot)
    batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(fileReader, batchSize=batchSize))
    for imageBatch, classBatch in batchGenerator:
        loss, acc = vgg.test(imageBatch, classBatch)
        result = (loss, acc)
        print("loss: %s acc: %f" % result)
        filesCount += batchSize
        successCount += (batchSize * acc)
    print ("total acc: %f in %d files" % (successCount/filesCount, filesCount))
