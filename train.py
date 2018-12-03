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
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')

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

# for graphs
cost_test = []
cost_train = []
acc_test = []
acc_train =[]

# training
for epoch in range(FLAGS.epochs):
    batch = next_batch(train_x, train_y, FLAGS.batch_size)
    batch_x = batch[0].reshape((-1, model.h_in, model.w_in, 1))
    batch_x = np.clip(batch_x, 0., 1.)
    batch_y = batch[1]

    batch_cost, batch_acc, _ = sess.run([model.cost, model.accuracy, model.opt_op], feed_dict={placeholders['inputs']:batch_x, placeholders['outputs']:batch_y})
    test_cost, test_acc = sess.run([model.cost, model.accuracy], feed_dict={placeholders['inputs']:test_x, placeholders['outputs']:test_y})

    cost_train.append(batch_cost)
    cost_test.append(test_cost)
    acc_train.append(batch_acc)
    acc_test.append(test_acc)

    if epoch % 30 == 0:
        print ('Epoch: {0}... cost {1}, test_cost {2}, acc {3}, test_acc {4}'.format(epoch, batch_cost, test_cost, batch_acc, test_acc))

    if epoch > FLAGS.early_stopping and cost_train[-1] > np.mean(cost_train[-(FLAGS.early_stopping+1):-1]):
        print ('Early stopping at epoch {}...'.format(epoch))
        print ('Epoch: {0}... cost {1}, test_cost {2}, acc {3}, test_acc {4}'.format(epoch, batch_cost, test_cost, batch_acc, test_acc))
        break

# show result
train_x = np.array(train_x).reshape((-1, model.h_in, model.w_in, 1))
train_x = np.clip(train_x, 0., 1.)
train_cost, train_acc = sess.run([model.cost, model.accuracy], feed_dict={placeholders['inputs']:train_x, placeholders['outputs']:train_y})
print ('training set: cost {0}, accuracy {1}'.format(train_cost, train_acc))
test_cost, test_acc = sess.run([model.cost, model.accuracy], feed_dict={placeholders['inputs']:test_x, placeholders['outputs']:test_y})
print ('test set: cost {0}, accuracy {1}'.format(test_cost, test_acc))

# save model
saver = tf.train.Saver()
saver.save(sess, 'data/models/cnn_model')

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='century'
plt.rcParams['font.size'] = 8
plt.figure(figsize=(4, 4))

plt.plot(range(len(cost_train)), cost_train, color = 'red', label = 'train', linewidth = 0.3)
plt.plot(range(len(cost_test)), cost_test, color = 'blue', label = 'test', linewidth = 0.3)
plt.title('Cost')
plt.ylabel('Mean Cross Entropy Error')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('data/training_result_cost.png', dpi=300)
plt.clf()

plt.plot(range(len(acc_train)), acc_train, color = 'red', label = 'train', linewidth = 0.3)
plt.plot(range(len(acc_test)), acc_test, color = 'blue', label = 'test', linewidth = 0.3)
plt.title('Accuracy')
plt.ylabel('Prediction Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('data/training_result_acc.png', dpi=300)
plt.clf()

sess.close()
