import logging
import os 
import numpy as np
import tensorflow as tf
from datetime import datetime
from layerutils import *
from preprocessing import data

#logging.basicConfig(format='%(asctime)s %(message)s',filename='cnn1.log', level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler())

def write_progress(epoch, iteration, learning_rate):
    f = open('/home/matthew/scifair/neuralnet/cnn1_progress.txt', 'a+')
    f.write(str(epoch) + '\n')
    f.write(str(iteration) + '\n')
    f.write(str(learning_rate) + '\n')
    f.close()

def write_train_accuracy(accuracy):
    f = open('/home/matthew/scifair/neuralnet/cnn1_train_accuracy.txt', 'a+')
    f.write(str(accuracy) + '\n')
    f.close()

def write_test_accuracy(accuracy):
    f = open('/home/matthew/scifair/neuralnet/cnn1_test_accuracy.txt', 'a+')
    f.write(str(accuracy) + '\n')
    f.close()

def get_progress():
    f = open('/home/matthew/scifair/neuralnet/cnn1_progress.txt', 'r')
    lines = f.readlines()
    epoch, iteration, learning_rate = lines[-3][:-1], lines[-2][:-1], lines[-1][:-1]
    return int(epoch), int(iteration), float(learning_rate)


### construction
tf.reset_default_graph()
num_training_examples = 11064
decay = 0.01
n = 1
X = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
training = tf.placeholder(tf.bool, name='training')
with tf.name_scope('cnn'):
    with tf.variable_scope('intro1'):
        conv1 = conv(X, 64, ksize=7, stride=2)
        max1 = max_pool(conv1)
    with tf.variable_scope('intro2'):
        conv2 = conv(max1, 64, ksize=7, stride=2)
    group1 = group(conv2, n, 'group1', downsample=False)
    group2 = group(group1, n, 'group2')
    group3 = group(group2, n, 'group3')
    avg1 = avg_pool(group3)
    flat1 = flatten(avg1)    
    logits = fc(flat1, 2, 'fc')
    test_softmax = tf.nn.softmax(logits)
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name='loss')
learning_rate = 0.1
with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    threshold = 1.0
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
    training_op = optimizer.apply_gradients(capped_gvs)
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 10
batch_size = 20
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir, now)
xentropy_summary = tf.summary.scalar('xentropy', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
#starting_epoch, starting_iteration, learning_rate = get_progress()

with tf.Session() as sess:
    logging.debug('Started!')
    init.run()
    if False:
        continue
    #if 'cnn1.ckpt.meta' in os.listdir('/home/matthew/scifair/neuralnet/'):
    #    saver.restore(sess, './cnn1.ckpt')
    #    for iteration in range(starting_iteration, num_training_examples // batch_size):
    #        X_batch, y_batch = data(iteration, batch_size)
    #        sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
    #        if iteration % 10 == 0:
    #            summary_str = xentropy_summary.eval(feed_dict={X: X_batch, y: y_batch, training: True})
    #            step = starting_epoch * (num_training_examples // batch_size) + iteration
    #            file_writer.add_summary(summary_str, step)
    #            save_path = saver.save(sess, '/home/matthew/scifair/neuralnet/cnn1.ckpt')
    #            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: True})
    #            logging.debug('iteration', iteration, 'accuracy', acc_train)
    #            logging.debug('iteration ' + str(iteration) + ' accuracy ' + str(acc_train))
    #            write_progress(starting_epoch, iteration, learning_rate)
    #            write_train_accuracy(acc_train)
    #    for epoch in range(starting_epoch + 1, n_epochs):
    #        for iteration in range(num_training_examples // batch_size):
    #            X_batch, y_batch = data(iteration, batch_size)
    #            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
    #            if iteration % 10 == 0:
    #                summary_str = xentropy_summary.eval(feed_dict={X: X_batch, y: y_batch, training: True})
    #                step = epoch * (num_training_examples // batch_size) + iteration
    #                file_writer.add_summary(summary_str, step)
    #                save_path = saver.save(sess, '/home/matthew/scifair/neuralnet/cnn1.ckpt')
    #                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: True})
    #                logging.debug('iteration', str(iteration), 'accuracy', str(acc_train))
    #                logging.debug('iteration ' + str(iteration) + ' accuracy ' + str(acc_train) )
    #                write_progress(epoch, iteration, learning_rate)
    #                write_train_accuracy(acc_train)
    #        learning_rate = learning_rate * 1/(1 + decay * epoch)
    #        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: False})
    #        test_features, test_labels = data(epoch, 100, 'test')
    #        acc_test = accuracy.eval(feed_dict={X: test_features, y: test_labels, training: False})
    #        write_test_accuracy(acc_test)
    #        logging.debug(str(epoch), 'Train accuracy:', str(acc_train), 'Test accuracy', acc_test)
    #        logging.debug('epoch ' + str(epoch) + ' Train accuracy: ' + str(acc_train) + ' Test accuracy ' + str(acc_test))
    else:
        for epoch in range(n_epochs):
            for iteration in range(3000 // batch_size):
                X_batch, y_batch = data(0, 20, '')
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
                if iteration % 10 == 0:
                    #summary_str = xentropy_summary.eval(feed_dict={X: X_batch, y: y_batch, training: True})
                    #step = epoch * (num_training_examples // batch_size) + iteration
                    #file_writer.add_summary(summary_str, step)
                    #save_path = saver.save(sess, '/home/matthew/scifair/neuralnet/cnn1.ckpt')
                    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: True})
                    print('epoch', epoch, 'iteration', iteration, 'acc_train', acc_train)
                    #logging.debug('iteration', str(iteration), 'accuracy', str(acc_train))
                    #logging.debug('iteration ' + str(iteration) + ' accuracy ' + str(acc_train) )
                    #write_progress(epoch, iteration, learning_rate)
                    #write_train_accuracy(acc_train)
            learning_rate = learning_rate * 1/(1 + decay * epoch)
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: False})
            #test_features, test_labels = data(epoch, 100, 'test')
            #acc_test = accuracy.eval(feed_dict={X: test_features, y: test_labels, training: False})
            #write_test_accuracy(acc_test)
            #logging.debug(epoch, 'Train accuracy:', acc_train, 'Test accuracy', acc_test)
            #logging.debug('epoch ' + epoch + ' Train accuracy: ' + acc_train + ' Test accuracy ' + acc_test)