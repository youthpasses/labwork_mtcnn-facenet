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

MODELS_BASE_DIR = 'models/facenet' # Directory where to write trained models and checkpoints.
GPU_MEMORY_FRACTION = 0.9

DATA_DIR = 'faces_align'
MODEL_DEF='nn4' # Model definition. Points to a module containing the definition of the inference graph.
MAX_NROF_EPOCHS = 1 # Number of epochs to run.
BATCH_SIZE = 128 # Number of images to process in a batch.
IMAGE_SIZE = 96 #Image size (height, width) in pixels.
EPOCH_SIZE = 100 # Number of batches per epoch.
EMBEDDING_SIZE = 128 # Dimensionality of the embedding.
KEEP_PROBABILITY = 1.0 # Keep probability of dropout for the fully connected layer(s).
WEIGHT_DECAY = 0.0 # L2 weight regularization.
DECOV_LOSS_FACTOR = 0.0 # DeCov loss factor.
CENTER_LOSS_FACTOR = 0.0 # Center loss factor.
CENTER_LOSS_ALFA = 0.95 # Center update rate for center loss.
OPTIMIZER = 'ADAGRAD' # The optimization algorithm to use, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']
LEARNING_RATE = 0.1 # Initial learning rate. If set to a negative value a learning rate, schedule can be specified in the file "learning_rate_schedule.txt
TRAIN_DATA_DIR = 'faces_align'
TEST_DATA_DIR = 'faces_check'


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
        images = np.array(images)
        images = images / 255.
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
    images = np.array(images)
    images = images / 255.
    return images

def getImageFromPath(imagepath):
    image = np.array(Image.open(imagepath))
    image.resize((IMAGE_SIZE, IMAGE_SIZE, 3))
    image = np.reshape(image, (IMAGE_SIZE *IMAGE_SIZE *3))
    return image






def main():
    # preprocessing
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(os.path.expanduser(MODELS_BASE_DIR), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # placeholder
    image_data_placeholder = tf.placeholder(tf.float32, [None, 96 * 96 * 3], name='image_data')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    # model
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

    with tf.Graph().as_default():





