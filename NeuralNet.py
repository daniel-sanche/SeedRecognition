from DataLoader import DataLoader, LoadFilesData
import tensorflow as tf
from math import sqrt
import numpy as np
from Visualization import visualizeImages, csvFromOutput
from enum import Enum
from os import walk, path, mkdir
import pandas as pd

class NetworkType(Enum):
    Generator=0,
    Discriminator=1

class NeuralNet(object):
    """"""
    """
    Neural Net Layers
    """
    def create_conv_layer(self, prev_layer, new_depth, prev_depth, trainable=True, name_prefix="conv", patch_size=3):
        W, b = self.create_variables([patch_size, patch_size, prev_depth, new_depth], [new_depth],
                                          name_prefix=name_prefix, trainable=trainable)
        new_layer = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(new_layer + b)

    def create_max_pool_layer(self, prev_layer):
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def create_fully_connected_layer(self, prev_layer, new_size, prev_size, trainable=True, name_prefix="fc"):
        W, b = self.create_variables([prev_size, new_size], [new_size], name_prefix=name_prefix,trainable=trainable)
        new_layer = tf.nn.relu(tf.matmul(prev_layer, W) + b)
        return new_layer

    def create_output_layer(self, prev_layer, prev_size, num_classes, trainable=True, name_prefix="out"):
        W, b = self.create_variables([prev_size, num_classes], [num_classes], name_prefix=name_prefix,trainable=trainable)
        return tf.nn.sigmoid(tf.matmul(prev_layer, W) + b)

    def create_deconv_layer(self, prev_layer, new_depth, prev_depth, trainable=True, name_prefix="deconv", patch_size=3, relu=True):
        input_shape = prev_layer.get_shape().as_list()
        new_shape = input_shape
        new_shape[-1] = new_depth
        W, b = self.create_variables([patch_size, patch_size, new_depth, prev_depth], [new_depth],
                                          name_prefix=name_prefix, trainable=trainable)
        new_layer = tf.nn.conv2d_transpose(prev_layer, W, new_shape, strides=[1, 1, 1, 1], padding='SAME')
        if relu:
            return tf.nn.relu(new_layer + b)
        else:
            return new_layer + b

    def create_variables(self, w_size, b_size, name_prefix="untitled", trainable=True, w_stddev=0.02, b_val=0.1):
        W_name = name_prefix + "-W"
        b_name = name_prefix + "-b"
        W = tf.Variable(tf.truncated_normal(w_size, stddev=w_stddev),
                        trainable=trainable, name=W_name)
        b = tf.Variable(tf.constant(b_val, shape=b_size), trainable=trainable, name=b_name)
        self.vardict[W_name] = W
        self.vardict[b_name] = b
        return W, b

    def create_batchnorm_layer(self, prev_layer, layer_shape, name_prefix="bnorm", trainable=True):
        scale_name = name_prefix + "-S"
        offset_name = name_prefix + "-O"
        mean, variance = tf.nn.moments(prev_layer, axes=[0])
        scale = tf.Variable(tf.ones(layer_shape), name=scale_name, trainable=trainable)
        offset = tf.Variable(tf.zeros(layer_shape), name=offset_name, trainable=trainable)
        self.vardict[scale_name] = scale
        self.vardict[offset_name] = offset
        return tf.nn.batch_normalization(prev_layer, mean, variance, offset, scale, 1e-8)

    def create_upsample_layer(self, prev_layer, new_size):
        resized = tf.image.resize_images(prev_layer, new_size, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return resized

    """
    Initialization Helpers
    """
    def __init__(self, trainingType, batch_size=1000, chkptDir="./checkpoints", chkptName="FaceGen.ckpt",image_size=64, noise_size=1000, age_range=[10, 100], learningRate=1e-4):
        self.vardict = {}
        self.trainingType = trainingType
        self.age_range = age_range
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_size=noise_size

        trainGen = (trainingType==NetworkType.Generator)
        self._buildGenerator(firstImageSize=8, train=trainGen)
        self._buildDiscriminator(conv1Size=64, conv2Size=32, fcSize=49, train=(not trainGen))
        self._buildCostFunctions(learningRate=learningRate)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        self.session = sess
        self.saver = tf.train.Saver(self.vardict, max_to_keep=3)
        self.checkpoint_name = chkptName
        self.checkpoint_dir = chkptDir
        self.checkpoint_num = 0
        self.restoreNewestCheckpoint()
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        pd.set_option('expand_frame_repr', False)

    def _buildGenerator(self, firstImageSize=8, train=True):

        # build the generator network
        gen_input_noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_size])
        gen_fully_connected1 = self.create_fully_connected_layer(gen_input_noise, firstImageSize*firstImageSize*64,
                                                                 self.noise_size,
                                                                 trainable=train, name_prefix="gen_fc")
        gen_squared_fc1 = tf.reshape(gen_fully_connected1, [self.batch_size, firstImageSize, firstImageSize, 64])
        gen_squared_fc1_norm = self.create_batchnorm_layer(gen_squared_fc1, [8,8,64], trainable=train, name_prefix="gen_fc")
        # s[1000,8,8,64]
        gen_unpool1 = self.create_upsample_layer(gen_squared_fc1_norm, 16)
        # [1000,16,16,64]
        gen_unconv1 = self.create_deconv_layer(gen_unpool1, 32, 64, trainable=train, name_prefix="gen_unconv1")
        gen_unconv1_norm = self.create_batchnorm_layer(gen_unconv1, [16, 16, 32], trainable=train,name_prefix="gen_unconv1")
        # [1000,16,16,32]
        gen_unpool2 = self.create_upsample_layer(gen_unconv1_norm, 32)
        # [1000,32,32,32]
        gen_unconv2 = self.create_deconv_layer(gen_unpool2, 16, 32, trainable=train, name_prefix="gen_unconv2")
        gen_unconv2_norm = self.create_batchnorm_layer(gen_unconv2, [32,32,16], trainable=train,name_prefix="gen_unconv2")
        # [1000,32,32,16]
        gen_unpool3 = self.create_upsample_layer(gen_unconv2_norm, 64)
        # [1000,64,64,16]
        gen_unconv3 = self.create_deconv_layer(gen_unpool3, 3, 16, trainable=train, name_prefix="gen_unconv3")
        gen_unconv3_norm = self.create_batchnorm_layer(gen_unconv3, [64,64,3], trainable=train,name_prefix="gen_unconv3")
        # [1000,64,64,3]
        gen_output_layer = tf.nn.tanh(gen_unconv3_norm)
        # [1000,12288]

        #save important nodes
        self.gen_output = gen_output_layer
        self.gen_input_noise = gen_input_noise
        self.gen0 = gen_squared_fc1
        self.gen1 = gen_unconv1
        self.gen2 = gen_unconv2
        self.gen3 = gen_unconv3

    def _buildDiscriminator(self, conv1Size, conv2Size, fcSize, train=True):
        dis_input_image = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
        #[1000, 64, 64, 3]

        dis_combined_inputs = tf.concat(0, [dis_input_image, self.gen_output])
        # [2000, 64, 64, 3]
        dis_conv1 = self.create_conv_layer(dis_combined_inputs, conv1Size, 3, trainable=train, name_prefix="dis_conv1")
        # [2000, 64, 64, 64]
        dis_pool1 = self.create_max_pool_layer(dis_conv1)
        # [2000, 32, 32, 64]
        dis_conv2 = self.create_conv_layer(dis_pool1, conv2Size, conv1Size, trainable=train, name_prefix="dis_conv2")
        # [2000, 32, 32, 32]
        dis_pool2 = self.create_max_pool_layer(dis_conv2)
        # [2000, 16, 16, 32]
        dis_pool2_flattened = tf.reshape(dis_pool2, [self.batch_size*2, -1])
        # [2000, 8192]
        dis_fully_connected1 = self.create_fully_connected_layer(dis_pool2_flattened, fcSize,
                                                                      16 * 16 * conv2Size,
                                                                      trainable=train,
                                                                      name_prefix="dis_fc")
        # [2000, 49]
        dis_output_layer = self.create_output_layer(dis_fully_connected1,fcSize,1,trainable=train,name_prefix="dis_out")
        # [2000, 3]

        # save important nodes
        self.dis_input_image = dis_input_image
        self.dis_output = dis_output_layer

    def _buildCostFunctions(self, learningRate=1e-4, beta1=0.5):
        generated_logits, true_logits = tf.split(0, 2, self.dis_output);

        self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(true_logits, tf.ones([self.batch_size, 1])))
        self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generated_logits, tf.zeros([self.batch_size, 1])))
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(generated_logits, tf.ones([self.batch_size, 1])))

        self.dis_vars = [self.vardict[varname] for varname in self.vardict if 'dis_' in varname]
        self.gen_vars = [self.vardict[varname] for varname in self.vardict if 'gen_' in varname]

        self.dis_train = tf.train.AdamOptimizer(learningRate, beta1=beta1).minimize(self.dis_loss, var_list=self.dis_vars)
        self.gen_train = tf.train.AdamOptimizer(learningRate, beta1=beta1).minimize(self.gen_loss, var_list=self.gen_vars)


    """
    Public Functions (Interface)
    """

    def saveCheckpoint(self, runsSinceLast):
        if runsSinceLast > 0:
            self.checkpoint_num = self.checkpoint_num + runsSinceLast
            self.saver.save(self.session, self.checkpoint_dir + "/" + self.checkpoint_name, self.checkpoint_num)
            print(self.checkpoint_name + " " + str(self.checkpoint_num) + " saved")


    def restoreNewestCheckpoint(self):
        if not path.exists(self.checkpoint_dir):
            mkdir(self.checkpoint_dir)
        highest_found = 0
        path_found = None
        for subdir, dirs, files in walk(self.checkpoint_dir):
            for file in files:
                if self.checkpoint_name in file and ".meta" not in file and ".txt" not in file:
                    iteration_num = int(file.split("-")[-1])
                    if iteration_num >= highest_found:
                        highest_found = iteration_num
                        path_found = path.join(subdir, file)
        if path_found is not None:
            #if existing one was found, restore previous checkpoint
            print ("restoring checkpoint ", path_found)
            self.saver.restore(self.session, path_found)
        else:
            print("no checkpoint found named " + self.checkpoint_name + " in " + self.checkpoint_dir)
        self.checkpoint_num = highest_found


    def _createFeedDict(self, truthImages, truthGenders, truthAges):
        truthImages.resize([self.batch_size, 64, 64, 3])
        batch_size = self.batch_size
        noise_batch = np.random.uniform(-1, 1, [batch_size, self.noise_size]).astype(np.float32)
        #ageVec = (np.linspace(start=self.age_range[0], stop=self.age_range[1],
        #                      num=batch_size) + np.random.sample(batch_size))
        #ageVec = ageVec.reshape([batch_size, 1]) / self.age_range[1]
        #ageVec = np.clip(ageVec, 0, 1)
        #genderVec = np.tile(np.array([0, 1], dtype=np.float64), int(batch_size / 2)).reshape([batch_size, 1])
        feed_dict = {self.gen_input_noise: noise_batch, self.dis_input_image: truthImages}
        return feed_dict

    def train(self, truthImages, truthGenders, truthAges):
        feed_dict = self._createFeedDict(truthImages, truthGenders, truthAges)
        #gen0, gen1, gen2, gen3 = self.session.run((self.gen0, self.gen1,self.gen2,self.gen3), feed_dict=feed_dict)
        if self.trainingType == NetworkType.Discriminator:
            self.session.run((self.dis_train), feed_dict=feed_dict)
        else:
            self.session.run((self.gen_train), feed_dict=feed_dict)

    def printStatus(self, truthImages, truthGenders, truthAges):
        feed_dict = self._createFeedDict(truthImages, truthGenders, truthAges)
        runList = (self.dis_loss_fake, self.dis_loss_real, self.gen_loss, self.gen_output)
        errFake, errReal, errGen, images = self.session.run(runList, feed_dict=feed_dict)
        print("d_loss: %.8f, g_loss: %.8f", (errFake + errReal), errGen)
        visualizeImages(images, numRows=10)



if __name__ == "__main__":
    #initialize the data loader
    datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
    csvPath = "./dataset.csv"
    indicesPath = "./indices.p"
    csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

    saveSteps = 10
    image_size = 64
    numPerBin = 10
    batch_size = numPerBin * 8 * 2
    loader = DataLoader(indices, csvdata, numPerBin=numPerBin, imageSize=image_size)
    loader.start()

    #start training
    discriminator = NeuralNet(trainingType=NetworkType.Discriminator, batch_size=batch_size, image_size=image_size, noise_size=20)
    i=0
    while True:
        batchDict = loader.getData()
        batchImage = batchDict["image"]
        batchAge = batchDict["age"]
        batchSex = batchDict["sex"]
        batchImage = batchImage.reshape([batch_size, -1])
        discriminator.train(batchImage, batchSex, batchAge, print_results=i%saveSteps==0)
        if i%saveSteps==0 and i != 0:
            discriminator.saveCheckpoint(saveSteps)
        i=i+1
