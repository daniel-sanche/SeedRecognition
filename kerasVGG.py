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
from flask import jsonify
import h5py
import os
import DataLoader
from sklearn.preprocessing import OneHotEncoder

# build the VGG16 network
class VGG:
    def __init__(self, numClasses=30):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
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

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        self.numClasses = numClasses


    # weights location: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    def loadWeights(self, weights_path):
        f = h5py.File(weights_path)
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

    def train(self, imageMat, LabelMat):
        oneHotLabels = np.zeros([imageMat.shape[0], self.numClasses])
        for i in range(LabelMat.shape[0]):
            oneHotLabels[i,LabelMat[i]] = 1
        loss = self.model.train_on_batch(imageMat, oneHotLabels)
        print("{}: {}: {} {}".format(self.model.metrics_names[0], loss[0], self.model.metrics_names[1], loss[1]))

    def predict(self, imageMat, probabilities=False):
        if probabilities:
            results = self.model.predict(imageMat, batch_size=imageMat.shape[0], verbose=0)
        else:
            results = self.model.predict_classes(imageMat, batch_size=imageMat.shape[0], verbose=0)
        print(results)
        return results


    def launch_server(self):
        server.launch(self.model, classes=[str(i) for i in range(30)])


if __name__ == "__main__":
    imageDir = "./GeneratedImages_Bin2"
    batchSize = 8

    vggModel = VGG()
    vggModel.loadWeights("vgg16_weights_keras.h5")
    #vggModel.launch_server()

    index = DataLoader.indexReader(os.path.join(imageDir, 'index.tsv'))
    for imageBatch, classBatch in DataLoader.batchLoader(imageDir, index, batchSize=batchSize):
        vggModel.train(imageBatch, classBatch.reshape([batchSize,]))
        #vggModel.predict(imageBatch, probabilities=True)


