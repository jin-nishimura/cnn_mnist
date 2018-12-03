# coding:utf-8
import tensorflow as tf
import time

from utils import *
from model import *

# settings
### tf.app.flags.DEFINE_string('variable', 'default', """explanation""")
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel') # for jupyter use
flags.DEFINE_string('dataset', 'mnist', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('epochs', 20000, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 126, 'Size of batch to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate.')
flags.DEFINE_integer('early_stopping', 2000, 'Tolerance for early stopping (# of epochs).')

# load data
train_x, train_y, test_x, test_y = load_data(FLAGS.dataset)
train_y, test_y = to_onehot(train_y), to_onehot(test_y)

# define placeholder
placeholders = {
    'inputs' : tf.placeholder(tf.float32, (None, train_x.shape[1], train_x.shape[2], 1), name = 'source'),
    'outputs' : tf.placeholder(tf.float32, (None, train_y.shape[1]), name = 'target')
}

# Create Model
model = ConvolutionalNetwork([train_x.shape[1], train_x.shape[2]], train_y.shape[1], placeholders)

sess = tf.Session()

# init variables
sess.run(tf.global_variables_initializer())
test_x = np.array(test_x).reshape((-1, model.h_in, model.w_in, 1))
test_x = np.clip(test_x, 0., 1.)

# laod model
saver = tf.train.Saver()
sess = tf.Session()
# Restore variables from disk.
saver.restore(sess, "data/models/cnn_model")
print("Model restored.")

train_x = np.array(train_x).reshape((-1, model.h_in, model.w_in, 1))
train_x = np.clip(train_x, 0., 1.)
train_cost, train_acc = sess.run([model.cost, model.accuracy], feed_dict={placeholders['inputs']:train_x, placeholders['outputs']:train_y})
print ('training set: cost {0}, accuracy {1}'.format(train_cost, train_acc))
test_cost, test_acc = sess.run([model.cost, model.accuracy], feed_dict={placeholders['inputs']:test_x, placeholders['outputs']:test_y})
print ('test set: cost {0}, accuracy {1}'.format(test_cost, test_acc))

test_pred = sess.run([model.correct_prediction], feed_dict={placeholders['inputs']:test_x, placeholders['outputs']:test_y})

train_x, train_y, test_x, test_y = load_data(FLAGS.dataset)

idx_false = np.where(np.array(test_pred).flatten() == 0)
wrong_pred = test_y[idx_false].flatten()
wrong = test_x[idx_false]
print(idx_false)
test_x = np.array(test_x).reshape((-1, model.h_in, model.w_in, 1))
test_x = np.clip(test_x, 0., 1.)
train_y, test_y = to_onehot(train_y), to_onehot(test_y)
logits = sess.run(model.logits, feed_dict={placeholders['inputs']:test_x, placeholders['outputs']:test_y})
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
#print(softmax(logits[67]),test_y[67])

import seaborn as sns
import matplotlib.pyplot as plt
idx = 0
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(8, 6))
for i in range(6):
    for j in range(6):
        sns.heatmap(wrong[idx], cmap = 'Greys', xticklabels = False, yticklabels = False, cbar = False, ax=axes[i, j])
        idx += 1
fig.savefig('data/digit_heatmap.png')
