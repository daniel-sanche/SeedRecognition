import DataLoader
from kerasVGG import VGG
import os
import numpy as np

if __name__ == "__main__":
    checkpointName = "./keras_checkpoint.h5"

    vgg = VGG()
    vgg.loadWeights(checkpointName, None)
    batchSize = 45

    filesCount = 0
    successCount = 0.0
    #segmentedReader = DataLoader.segmentedDatasetParser("/home/sanche/Datasets/Seed_Test")
    generatedReader = DataLoader.generatedDatasetParser("./GeneratedImages_Bin2")
    batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(generatedReader, batchSize=batchSize))
    for imageBatch, classBatch in batchGenerator:
        loss, acc = vgg.test(imageBatch, classBatch)
        result = (loss, acc)
        filesCount += batchSize
        successCount += (batchSize * acc)
    print ("total acc: %f in %d files" % (successCount/filesCount, filesCount))
