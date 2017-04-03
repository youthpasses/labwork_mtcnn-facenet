# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import Image
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


pretrained_model = './models/20170401-161132/model-20170401-161132.ckpt-400' # Load a pretrained model before training starts.
pretrained_model_dir = './models/20170401-161132'
data_dir = 'faces_align'
model_def ='nn4'
batch_size = 128 # Number of images to process in a batch.
image_size = 96
keep_probability = 0.9
embedding_size = 128
random_rotate = True
random_crop = True
random_flip = True

TRAIN_DATA_DIR = 'faces_align'
TEST_DATA_DIR = 'faces_check'

def getTrainPathsDic():
    train_set = facenet.get_dataset(TRAIN_DATA_DIR)
    train_dic = {}
    for imageClass in train_set:
        className = imageClass.name
        imagePaths = imageClass.image_paths
        imagePaths = [[imagepath] for imagepath in imagePaths]
        train_dic[className] = imagePaths
        print(className)
        print(imagePaths)
    return train_dic


def getTestPaths():
    test_paths = []
    path_exp = os.path.expanduser(TEST_DATA_DIR)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        test_paths = [os.path.join(path_exp, image) for image in images]
    return test_paths


def main():
    network = importlib.import_module(model_def)
    tf.Graph().as_default()
    sess = tf.Session()

    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
    # phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    batch_norm_params = {
        # Decay for the moving averages
        'decay': 0.995,
        # epsilon to prevent 0s in variance
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        # Only update statistics during training mode
        'is_training': False
    }
    input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                          dtypes=[tf.string],
                                          shapes=[(1,)],
                                          shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder], name='enqueue_op')
    filenames = input_queue.dequeue()
    images = []
    for filename in tf.unstack(filenames):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents)
        if random_rotate:
            image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)

        # pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        images.append(tf.image.per_image_standardization(image))
    # images = np.array(images)
    print("image.shape = ", images[0].shape)

    image_batch = tf.train.batch(
        images,
        batch_size=batch_size,
        enqueue_many=True,
        capacity=batch_size,
        allow_smaller_final_batch=True)
    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')

    prelogits, _ = network.inference(image_batch, keep_probability,
                                     phase_train=False, weight_decay=0.0)
    bottleneck = slim.fully_connected(prelogits, embedding_size, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.0),
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params=batch_norm_params,
                                      scope='Bottleneck', reuse=False)
    embeddings = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='embeddings')

    # ema = tf.train.ExponentialMovingAverage(1.0)
    # saver = tf.train.Saver(ema.variables_to_restore())
    # ckpt = tf.train.get_checkpoint_state(os.path.expanduser(pretrained_model_dir))
    # saver.restore(sess, ckpt.model_checkpoint_path)
    meta_file, ckpt_file = facenet.get_model_filenames(pretrained_model_dir)
    facenet.custom_load_model(sess, pretrained_model_dir, meta_file, ckpt_file)


    trainpathdic = getTrainPathsDic()
    testpath = getTestPaths()
    for classname in trainpathdic.keys():
        print("classname", classname)
        classpaths = trainpathdic[classname]
        sess.run([enqueue_op, image_batch], {image_paths_placeholder: classpaths})
        sess.run([prelogits, _])
        sess.run(bottleneck)
        emb_train = sess.run(embeddings)
        print('emb_train.shape', emb_train.shape)
        print(emb_train.eval())


main()




'''

LOGS_BASE_DIR = 'logs/facenet'  # Directory where to write event logs.
MODELS_BASE_DIR = 'models/facenet'  # Directory where to write trained models and checkpoints.
GPU_MEMORY_FRACTION = 0.9  # Upper bound on the amount of GPU memory that will be used by the process.
PRETRAINED_MODEL = './models/facenet/20170330-224911/model-20170330-224911.ckpt-2500'  # Load a pretrained model before training starts.
PRETRAINED_MODEL_DIR = './models/facenet/20170330-224911'
#     parser.add_argument('--data_dir', type=str,
#         help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
#         default='~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned')
TRAIN_DATA_DIR = 'faces_align'
MODEL_DEF = 'nn4'  # Model definition. Points to a module containing the definition of the inference graph.
MAX_NROF_EPOCHS = 500  # Number of epochs to run.
BATCH_SIZE = 128  # Number of images to process in a batch.
IMAGE_SIZE = 96  # Image size (height, width) in pixels.
EPOCH_SIZE = 1000  # Number of batches per epoch.
EMBEDDING_SIZE = 128  # Dimensionality of the embedding.
KEEP_PROBABILITY = 1.0  # Keep probability of dropout for the fully connected layer(s).
WEIGHT_DECAY = 0.0  # L2 weight regularization.
DECOV_LOSS_FACTOR = 0.0  # DeCov loss factor.
CENTER_LOSS_FACTOR = 0.0  # Center loss factor.
CENTER_LOSS_ALFA = 0.95  # Center update rate for center loss.
OPTIMIZER = 'ADAGRAD'  # The optimization algorithm to use, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']
LEARNING_RATE = 0.1  # Initial learning rate. If set to a negative value a learning rate, schedule can be specified in the file "learning_rate_schedule.txt
LEARING_RATE_DECAY_EPOCHS = 100  # Number of epochs between learning rate decay.
LEARING_RATE_DECAY_FACTOR = 1.0  # Learning rate decay factor.
MOVING_AVERAGE_DECAY = 0.9999  # Exponential decay for tracking of training parameters.


RANDOM_CROP = True  # Performs random cropping of training images. If false, the center image_size pixels from the training images are used. \
#  If the size of the images in the data directory is equal to image_size no cropping is performed
RANDOM_FLIP = True  # Performs random horizontal flipping of training images.
RANDOM_ROTATE = True  # Performs random rotations of training images.
LOG_HISTOGRAMS = True  # Enables logging of weight/bias histograms in tensorboard.
NO_STORE_REVISION_INFO = True  # Disables storing of git revision info in revision_info.txt.
PHASE_TRAIN = False


TRAIN_DATA_DIR = 'faces_align'
TEST_DATA_DIR = 'faces_check'



def getTrainPaths():
    train_set = facenet.get_dataset(TRAIN_DATA_DIR)
    train_dic = {}
    for imageClass in train_set:
        className = imageClass.name
        imagePaths = imageClass.image_paths
        imagePaths = [[imagepath] for imagepath in imagePaths]
        train_dic[className] = imagePaths
        print(className)
        print(imagePaths)
    return train_dic


def getTestPaths():
    test_paths = []
    path_exp = os.path.expanduser(TEST_DATA_DIR)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        test_paths = [os.path.join(path_exp, image) for image in images]
    return test_paths

def getTrainData():
    train_set = facenet.get_dataset(TRAIN_DATA_DIR)
    train_dic = {}
    for imageClass in train_set:
        className = imageClass.name
        imagePaths = imageClass.image_paths
        images = []
        for imagepath in imagePaths:
            image = getImageFromPath(imagepath)
            # print(image)
            images.append(image)
        images = np.array(images)
        train_dic[className]= images
    return train_dic

def getTestData():
    images = []
    path_exp = os.path.expanduser(TEST_DATA_DIR)
    if os.path.isdir(path_exp):
        imagenames = os.listdir(path_exp)
        for imagename in imagenames:
            imagepath = os.path.join(path_exp, imagename)
            image = getImageFromPath(imagepath)
            images.append(image)
    return images

def getImageFromPath(imagepath):
    image = np.array(Image.open(imagepath))
    image.resize((IMAGE_SIZE, IMAGE_SIZE, 3))
    # image = tf.image.per_image_standardization(image)
    # print("image.stand.shape = "image.shape)
    image = np.reshape(image, (IMAGE_SIZE *IMAGE_SIZE *3))
    return image
    # file_contents = tf.read_file(imagepath)
    # image = tf.image.decode_png(file_contents)
    # if RANDOM_ROTATE:
    #     image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
    # if RANDOM_CROP:
    #     image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    # else:
    #     image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    # if RANDOM_FLIP:
    #     image = tf.image.random_flip_left_right(image)
    # # pylint: disable=no-member
    # image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    # return tf.image.per_image_standardization(image)



def main():
    image_data_placeholder = tf.placeholder(tf.float32, [None, 96 * 96 * 3], name='image_data')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    network = importlib.import_module(MODEL_DEF)
    batch_norm_params = {
        # Decay for the moving averages
        'decay': 0.995,
        # epsilon to prevent 0s in variance
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        # Only update statistics during training mode
        'is_training': phase_train_placeholder
    }
    # Build the inference graph
    prelogits, net = network.inference(image_data_placeholder, KEEP_PROBABILITY,
                                       phase_train=phase_train_placeholder, weight_decay=WEIGHT_DECAY)
    bottleneck = slim.fully_connected(prelogits, EMBEDDING_SIZE, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params=batch_norm_params,
                                      scope='Bottleneck', reuse=False)
    embeddings = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='embeddings')

    pretrained_model = None
    if PRETRAINED_MODEL:
        pretrained_model = os.path.expanduser(PRETRAINED_MODEL)
        print('训练ok模型: Trained model: %s' % pretrained_model)
    else:
        print('没有训练ok模型...')


    print(image_data_placeholder.shape)

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # saver.restore(sess, pretrained_model)
    ckpt = tf.train.get_checkpoint_state(PRETRAINED_MODEL_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring the trained model: %s' % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    with sess.as_default():

        # get train and test data
        train_dic = getTrainData()
        test_data = getTestData()
        emb_dic = {}
        emb_mean_dic = {}
        for className in train_dic.keys():
            image_data = train_dic[className]
            print("image_data.shape", image_data.shape)

            prelogits, _, emb = sess.run([prelogits, net, embeddings], feed_dict={image_data_placeholder: image_data, phase_train_placeholder:PHASE_TRAIN})
            print('\noutput embeddings')
            print(emb)
            print(emb.get_shape())
            emb_dic[className] = emb
            emb_mean = np.mean(emb, axis=0)
            print(emb_mean)
            emb_mean_dic[className] = emb_mean
        for className in emb_mean_dic.keys():
            print(className)
            print(emb_mean_dic[className])
            print('\n')

        print('Train Finished!')
        prelogits_test, _, emb_test = sess.run([prelogits, net, embeddings], feed_dict={image_data_placeholder: test_data,
                                                                              phase_train_placeholder: PHASE_TRAIN})
        print('emb_test.shape')
        print(emb_test.shape)
        print(emb_test)

    sess.close()



def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold

def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset


main()
'''