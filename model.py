# coding:utf-8
import numpy as np
import tensorflow as tf

from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class ConvolutionalNetwork:
    def __init__(self, input_dims, output_dim, placeholders):
        self.input_dims = input_dims # [N, M]
        self.output_dim = output_dim
        self.placeholder = placeholders

        self.hyperparams = {
            'kernel' : (2, 2),
            'pooling': (2, 2),
            'stride' : (2, 2),
            'filter' : [32, 32, 16],
            'activation' : tf.nn.relu
        }

        self.full = self.input_dims[0] * self.input_dims[1]
        self.h_in = self.input_dims[0]
        self.w_in = self.input_dims[1]
        self.h_2  = int(np.ceil(float(self.h_in)/float(self.hyperparams['stride'][0])))
        self.w_2  = int(np.ceil(float(self.w_in)/float(self.hyperparams['stride'][0])))
        self.h_3  = int(np.ceil(float(self.h_2)/float(self.hyperparams['stride'][0])))
        self.w_3  = int(np.ceil(float(self.w_2)/float(self.hyperparams['stride'][0])))

        self.inputs = placeholders['inputs']
        self.outputs = placeholders['outputs']
        self.logits = None

        self.layers = {}
        self.loss = None
        self.cost = None
        self.accuracy  = None
        self.correct_prediction = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = None

        self._build()

    def _loss(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.outputs, logits = self.logits)

    def _cost(self):
        self.cost = tf.reduce_mean(self.loss)

    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.outputs, 1)), tf.float32))

    def _prediction(self):
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.outputs, 1)), tf.float32)

    def _build(self):
        self.layers.update({'my_layer': my_layer(self.inputs, self.hyperparams, self.input_dims, self.output_dim)})

        self.logits = self.layers['my_layer'].fully_c2
        self._loss()
        self._cost()
        self._accuracy()
        self._prediction()
        self.opt_op = self.optimizer.minimize(self.cost)
