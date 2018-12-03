# coding:utf-8
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

class my_layer:
    def __init__(self, inputs, pl, input_dim, output_dim):
        self.conv1 = tf.layers.conv2d(inputs, pl['filter'][0], pl['kernel'], padding = 'same', activation = pl['activation'])
        self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pl['pooling'], pl['stride'], padding = 'same')
        self.conv2 = tf.layers.conv2d(self.maxpool1, pl['filter'][1], pl['kernel'], padding = 'same', activation = pl['activation'])
        self.maxpool2 = tf.layers.max_pooling2d(self.conv2, pl['pooling'], pl['stride'], padding = 'same')
        self.conv3 = tf.layers.conv2d(self.maxpool2, pl['filter'][2], pl['kernel'], padding = 'same', activation = pl['activation'])
        self.conv3_flatten = tf.contrib.layers.flatten(self.conv3)
        self.fully_c1 = tf.contrib.layers.fully_connected(self.conv3_flatten, input_dim[0]*input_dim[1])
        self.dropout = tf.nn.dropout(self.fully_c1, FLAGS.dropout)
        self.fully_c2 = tf.contrib.layers.fully_connected(self.dropout, output_dim)
