from DataLoader import DataLoader, LoadFilesData
import tensorflow as tf
from math import sqrt, ceil
import numpy as np
from nnLayers import create_fully_connected_layer, create_max_pool_layer, create_deconv_layer, create_conv_layer, create_output_layer, create_unpool_layer
from Visualization import visualizeImages


class NeuralNet(object):
    def __init__(self, batch_size=1000, image_size=64, noise_size=20, age_range=[10, 100]):
        self.age_range = age_range
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_size=noise_size

        self.dropout =  tf.placeholder(tf.float32)
        self._buildGenerator(fcSize=64)
        self._buildDiscriminator(conv1Size=32, conv2Size=64, fcSize=49)
        self._buildDiscriminatorCost()

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        self.session = sess

    def _buildGenerator(self, fcSize):
        sqrtFc = int(sqrt(fcSize))

        # build the generator network
        gen_input_noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_size])
        gen_input_age = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        gen_input_gender = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        gen_input_combined = tf.concat(1, [gen_input_age, gen_input_gender, gen_input_noise])
        gen_fully_connected1, var_dict = create_fully_connected_layer(gen_input_combined, fcSize,
                                                                      self.noise_size + 2,
                                                                      self.dropout, trainable=True, name_prefix="gen_fc")
        gen_squared_fc1 = tf.reshape(gen_fully_connected1, [self.batch_size, sqrtFc, sqrtFc, 1])
        # now [1000,8,8,1]
        gen_unpool1 = create_unpool_layer(gen_squared_fc1)
        # now [1000,16,16,1]
        gen_unconv1, var_dict = create_deconv_layer(gen_unpool1, 5, 1, trainable=True, name_prefix="gen_unconv1",
                                                    var_dict=var_dict)
        # now [1000,16,16,5]
        gen_unpool2 = create_unpool_layer(gen_unconv1)
        # now [1000,32,32,5]
        gen_unconv2, var_dict = create_deconv_layer(gen_unpool2, 5, 5, trainable=True, name_prefix="gen_unconv2",
                                                    var_dict=var_dict)
        # now [1000,32,32,5]
        gen_unpool3 = create_unpool_layer(gen_unconv2)
        # now [1000,64,64,5]
        gen_unconv3, var_dict = create_deconv_layer(gen_unpool3, 3, 5, trainable=True, name_prefix="gen_unconv3",
                                                    var_dict=var_dict)
        # now [1000,64,64,5]
        totalPixels = self.image_size * self.image_size * 3
        gen_output_layer = tf.reshape(gen_unconv3, [self.batch_size, totalPixels])
        # now [1000,64,64,3]

        #save important nodes
        self.gen_output = gen_output_layer
        self.gen_input_noise = gen_input_noise
        self.gen_input_age = gen_input_age
        self.gen_input_gender = gen_input_gender
        self.vardict = var_dict

    def _buildDiscriminator(self, conv1Size, conv2Size, fcSize):
        num_pixels = self.image_size * self.image_size * 3
        dis_input_image = tf.placeholder(tf.float32, shape=[self.batch_size, num_pixels])
        dis_input_age = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        dis_input_gender = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        dis_labels_truth = tf.concat(0, [tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])])
        dis_labels_age = tf.concat(0, [dis_input_age, self.gen_input_age])
        dis_labels_gender = tf.concat(0, [dis_input_gender, self.gen_input_gender])
        #[1000, 12,288]

        dis_combined_inputs = tf.concat(0, [dis_input_image, self.gen_output])
        # [2000, 12288]
        dis_reshaped_inputs = tf.reshape(dis_combined_inputs, [self.batch_size * 2, self.image_size, self.image_size, 3])
        # [2000, 64, 64, 3]
        dis_conv1, var_dict = create_conv_layer(dis_reshaped_inputs, conv1Size, 3, trainable=True,
                                                name_prefix="dis_conv1", var_dict=self.vardict)
        # [2000, 64, 64, 32]
        dis_pool1 = create_max_pool_layer(dis_conv1)
        # [2000, 32, 32, 32]
        dis_conv2, var_dict = create_conv_layer(dis_pool1, conv2Size, conv1Size, trainable=True,
                                                name_prefix="dis_conv2", var_dict=var_dict)
        # [2000, 32, 32, 64]
        dis_pool2 = create_max_pool_layer(dis_conv2)
        # [2000, 16, 16, 64]
        dis_pool2_flattened = tf.reshape(dis_pool2, [self.batch_size*2, -1])
        # [2000, 16384]
        dis_fully_connected1, var_dict = create_fully_connected_layer(dis_pool2_flattened, fcSize,
                                                                      16 * 16 * conv2Size, self.dropout,
                                                                      trainable=True,
                                                                      name_prefix="dis_fc", var_dict=var_dict)
        # [2000, 49]
        dis_output_layer, var_dict = create_output_layer(dis_fully_connected1, fcSize, 3,
                                                         trainable=True, name_prefix="dis_out",
                                                         var_dict=var_dict)
        # [2000, 3]
        # save important nodes
        self.dis_input_image = dis_input_image
        self.dis_input_age = dis_input_age
        self.dis_input_gender = dis_input_gender
        self.dis_output = dis_output_layer
        self.vardict = var_dict
        self.dis_label_truth = dis_labels_truth
        self.dis_label_age = dis_labels_age
        self.dis_label_gender = dis_labels_gender


    def _buildDiscriminatorCost(self, learningRate=1e-4):
        isGenerated, age, gender = tf.split(1, 3, self.dis_output)
        scaledAge = tf.scalar_mul(self.age_range[1], age)
        truthDiff = tf.sub(isGenerated, self.dis_label_truth)
        ageDiff = tf.sub(scaledAge, self.dis_label_age)
        genderDiff = tf.sub(scaledAge, self.dis_label_gender)
        truthCost = tf.nn.l2_loss(truthDiff) / batch_size * 2
        genderCost = tf.nn.l2_loss(genderDiff) / batch_size * 2
        ageCost = tf.nn.l2_loss(ageDiff) / batch_size * 2
        combinedCost = tf.add(tf.add(truthCost, genderCost), ageCost)
        training_step = tf.train.AdamOptimizer(learningRate).minimize(combinedCost)
        self.dis_train = training_step


    def train(self, truthImages, truthGenders, truthAges):
        batch_size = self.batch_size
        noise_batch = np.random.random_sample((batch_size, self.noise_size))
        ageVec = (np.linspace(start=self.age_range[0], stop=self.age_range[1], num=batch_size) + np.random.sample(batch_size)).reshape([batch_size, 1])
        genderVec = np.tile(np.array([0, 1], dtype=bool), int(batch_size / 2)).reshape([batch_size, 1])
        feed_dict = {self.gen_input_noise: noise_batch, self.gen_input_age: ageVec,
                     self.gen_input_gender: genderVec, self.dropout: 0.5, self.dis_input_gender:truthGenders,
                     self.dis_input_age:truthAges, self.dis_input_image:truthImages}
        self.session.run(self.dis_train, feed_dict=feed_dict)
        #generatedImages = self.session.run(self.gen_output, feed_dict=feed_dict)
        #generatedImages = np.reshape(generatedImages, [batch_size, self.image_size, self.image_size, 3])
        #visualizeImages(generatedImages[:50, :, :, :], numRows=5)



#initialize the data loader
datasetDir = "/Users/Sanche/Datasets/IMDB-WIKI"
csvPath = "./dataset.csv"
indicesPath = "./indices.p"
csvdata, indices = LoadFilesData(datasetDir, csvPath, indicesPath)

image_size = 64
batch_goal = 1000
loader = DataLoader(indices, csvdata, batchSize=batch_goal, imageSize=image_size)
loader.start()
batchDict = loader.getData()
batchImage = batchDict["image"]
batchAge = batchDict["age"]
batchSex = batchDict["sex"]
batch_size = batchImage.shape[0]
batchImage = batchImage.reshape([batch_size, -1])

#start training
network = NeuralNet(batch_size=batch_size, image_size=image_size, noise_size=20)
network.train(batchImage, batchSex, batchAge)
print("done")