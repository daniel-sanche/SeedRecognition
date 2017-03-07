import DataLoader
from kerasVGG import VGG
import os
import numpy as np

if __name__ == "__main__":
    checkpointName = "./keras_checkpoint.h5"
    genPath = "./GeneratedImages_Bin2"
    segPath = "/home/sanche/Datasets/Seed_Test_Segmented"
    testPath = "/home/sanche/Datasets/Seed_Test"

    vgg = VGG()
    vgg.loadWeights(checkpointName, None)
    dataset = testPath

    if dataset == genPath:
        fileCount = len([name for name in os.listdir(dataset) if ".png" in name])
        parser = DataLoader.generatedDatasetParser(dataset)
    elif dataset == segPath:
        fileCount = 270
        parser = DataLoader.segmentedDatasetParser(dataset)
    else:
        fileCount = 1400
        parser = DataLoader.testDatasetParser(dataset)

    batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(parser, batchSize=1))
    vgg.test(batchGenerator, fileCount)
