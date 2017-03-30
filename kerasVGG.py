'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes (with the TensorFlow backend).
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from quiver_engine import server
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from flask import jsonify
import h5py
import os
import DataLoader
from keras import metrics
from keras.utils.np_utils import to_categorical
import functools
from math import ceil
import pandas as pd

# build the VGG16 network
class VGG:
    def __init__(self, imageSize=[224,224,3], numClasses=30):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=imageSize))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numClasses, activation='softmax'))

        #opt = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        opt = optimizers.SGD(lr=1e-4)
        top3 = functools.partial(metrics.top_k_categorical_accuracy, k=3)
        top3.__name__ = 'top3'
        top5 = functools.partial(metrics.top_k_categorical_accuracy, k=5)
        top5.__name__ = 'top5'
        top10 = functools.partial(metrics.top_k_categorical_accuracy, k=10)
        top10.__name__ = 'top10'

        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy',top3,top5, top10])
        self.model = model
        self.numClasses = numClasses


    # weights location: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    def loadWeights(self, checkpointWeights, baseWeights):
        if os.path.exists(checkpointWeights):
            print("loading from checkpoint...")
            self.model.load_weights(checkpointWeights)
        else:
            print("loading from vgg base...")
            f = h5py.File(baseWeights)
            for k in range(f.attrs['nb_layers']):
                if k >= len(self.model.layers) - 1:
                    # we don't look at the last two layers in the savefile (fully-connected and activation)
                    break
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                layer = self.model.layers[k]
                if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
                    weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

                layer.set_weights(weights)
            f.close()
        print('Model loaded.')

    def save(self, path):
        print("saving checkpoint...")
        self.model.save_weights(path, True)

    def train(self, batchGenerator, epochSize=100000,numEpochs=1):
        self.model.fit_generator(batchGenerator, samples_per_epoch=epochSize, nb_epoch=numEpochs, verbose=1)

    def predict(self, imageMat, probabilities=False):
        if probabilities:
            results = self.model.predict(imageMat, batch_size=imageMat.shape[0], verbose=0)
        else:
            results = self.model.predict_classes(imageMat, batch_size=imageMat.shape[0], verbose=0)
        return results

    def test(self, datasetGenerator, numImages):
        batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(datasetGenerator, batchSize=1))
        results = self.model.evaluate_generator(batchGenerator, numImages)
        print("testing complete")
        for i in range(len(results)):
            print("\t{}: {}".format(self.model.metrics_names[i], results[i]))

    def covarianceMatrix(self, datasetGenerator, numImages, savePath="./covariance_matrix.csv"):
        batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(datasetGenerator, batchSize=16))
        numBatches = int(ceil(numImages/16.0))
        resultDict = {}
        for i in range(numBatches):
            imgMat, labMat = next(batchGenerator)
            predMat = self.predict(imgMat, False)
            for j in range(16):
                truth = np.argmax(labMat[j])
                guess = predMat[j]
                oldInner = resultDict.get(truth, {})
                oldInner[guess] = oldInner.get(guess, 0) + 1
                resultDict[truth] = oldInner
        df = pd.DataFrame.from_dict(resultDict)
        df.index.name = "predictions"
        df.to_csv(savePath)
        return df

    def topNPerClass(self, datasetGenerator, numImages, nList=[1,3,5], savePath="./topN.csv"):
        batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(datasetGenerator, batchSize=16))
        numBatches = int(ceil(numImages / 16.0))
        successCount = {}
        totalCount = {}
        percentCount = {}
        for i in range(numBatches):
            imgMat, labMat = next(batchGenerator)
            predMat = self.predict(imgMat, True)
            for j in range(16):
                truth = np.argmax(labMat[j])
                for n in nList:
                    thisNSuccess = successCount.get(n, {})
                    topN = np.argpartition(-predMat[j], n)[:n]
                    if truth in topN:
                        thisNSuccess[truth] = thisNSuccess.get(truth, 0) + 1
                        successCount[n] = thisNSuccess
                totalCount[truth] = totalCount.get(truth, 0) + 1
        #find the percentages
        for key in totalCount:
            for n in nList:
                thisNPercent = percentCount.get(n, {})
                thisNPercent[key] = float((successCount.get(n, {})).get(key, 0.0)) / totalCount.get(key, 0.0)
                percentCount[n] = thisNPercent
        df = pd.DataFrame.from_dict(percentCount)
        df.to_csv(savePath)
        return df




    def launch_server(self, imgFolder="GeneratedImages_Bin2"):
        names = ['bc', 'bj', 'bn', 'sa', 'bry', 'brb',  'cm', 'cst', 'cso', 'sl',\
                 'cbp', 'brc', 'cd', 'ds', 'brp', 'sf1', 'sii1', 'siv1', 'sp1', 'sv1',\
                 'cca', 'cch', 'cgr', 'cme', 'cpe', 'ahy', 'apacc', 'apr', 'apo', 'are']
        server.launch(self.model, classes=names, input_folder=imgFolder)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    datasetDir = "./New_Generated"
    datasetSize = len([name for name in os.listdir(datasetDir) if ".png" in name])
    checkpointName = "./keras_checkpoint.h5"
    baseName = "vgg16_weights_keras.h5"
    batchSize = 8
    saveInterval = 50

    vggModel = VGG()
    vggModel.loadWeights(checkpointName, baseName)
    #vggModel.launch_server("/home/sanche/Datasets/Seed_Test/p1_45_first")

    i=0
    parser = DataLoader.generatedDatasetParser(datasetDir)
    batchGenerator = DataLoader.oneHotWrapper(DataLoader.batchLoader(parser, batchSize=batchSize))

    while True:
        vggModel.train(batchGenerator, epochSize=datasetSize)
        vggModel.save(checkpointName)
