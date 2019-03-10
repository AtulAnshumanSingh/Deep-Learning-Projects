# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:35:12 2019

@author: u346442
"""

import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class CNN:
    
    def __init__(self,LEARNING_RATE,BATCH_SIZE,DROPOUT,N_EPOCHS,N_CLASSES):
        
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.SKIP_STEP = SKIP_STEP
        self.DROPOUT = DROPOUT
        self.N_EPOCHS = N_EPOCHS
        self.N_CLASSES = N_CLASSES
        
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
            self.Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")
            
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def _AddConv2d(self, prev_layer, kernel_size, stride, bias_size,scope_name):
        
        with tf.variable_scope(scope_name) as scope:
            kernel = tf.get_variable('kernel', kernel_size, initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', bias_size, initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(prev_layer, kernel, strides=stride, padding='SAME')
            conv1 = tf.nn.relu(conv + biases, name=scope.name)
        
        return conv1
        
    def _AddAvgPool(self, prev_layer, ksize, strides, scope_name):
        
        with tf.variable_scope(scope_name) as scope:
            pool = tf.nn.max_pool(prev_layer, ksize = ksize, strides = strides,padding='SAME')
        
        return pool
    
    def _AddFullyConnected(self, prev_layer, input_feature, scope_name):
            
        with tf.variable_scope(scope_name):
            input_features = input_feature
            w = tf.get_variable('weights', [input_features, 1024], initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [1024],initializer=tf.constant_initializer(0.0))
        
            pool = tf.reshape(prev_layer, [-1, input_features])
            fc = tf.nn.relu(tf.matmul(pool, w) + b, name='relu')
        
            fc = tf.nn.dropout(fc, self.dropout, name='relu_dropout')
        
        return fc
    
    def _AddSoftmax(self, FullyConnectedLayer):
        
        with tf.variable_scope('softmax_linear') as scope:
            w = tf.get_variable('weights', [1024, self.N_CLASSES], initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [self.N_CLASSES], initializer=tf.random_normal_initializer())
            logits = tf.matmul(FullyConnectedLayer, w) + b
        
        return logits
    
    def _CreateLoss(self, Y, logits):
                
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
            loss = tf.reduce_mean(entropy, name='loss')
        
        return entropy, loss
    
    def _CreateOptimizer(self,loss):
        
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss, global_step=self.global_step)
        
        return optimizer
    
    def _BuildGraph(self):
        
        images = tf.reshape(self.X, shape=[-1, 28, 28, 1]) 
        
        self.graph = {}
        
        #Define architecture here:
        self.graph['conv1']   = self._AddConv2d(images, [5, 5, 1, 32], [1,1,1,1], [32], 'conv1')
        self.graph['pool1']   = self._AddAvgPool(self.graph['conv1'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool1')
        self.graph['conv2']   = self._AddConv2d(self.graph['pool1'], [5, 5, 32, 64], [1,1,1,1], [64], 'conv2')
        self.graph['pool2']   = self._AddAvgPool(self.graph['conv2'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool2')
        self.graph['fc']      = self._AddFullyConnected(self.graph['pool2'], 7 * 7 * 64, 'fc')
        self.graph['softmax'] = self._AddSoftmax(self.graph['fc'])
        self.graph['entropy'], self.graph['loss']  = self._CreateLoss(self.Y, self.graph['softmax'])
        
        # create optimizer
        self.graph['optimizer'] = self._CreateOptimizer(self.graph['loss'])
        
    
def train(MNIST, model):
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        initial_step = model.global_step.eval()
    
        start_time = time.time()
        n_batches = int(MNIST.train.num_examples / model.BATCH_SIZE)
    
        total_loss = 0.0
        for index in range(initial_step, n_batches * model.N_EPOCHS):
            X_batch, Y_batch = MNIST.train.next_batch(model.BATCH_SIZE)
            _, loss_batch = sess.run([model.graph['optimizer'], model.graph['loss']], 
                                    feed_dict={model.X: X_batch, model.Y:Y_batch, model.dropout: model.DROPOUT}) 
            total_loss += loss_batch
            if (index + 1) % model.SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / model.SKIP_STEP))
                total_loss = 0.0
        
        print("Optimization Finished!")
        print("Total time: {0} seconds".format(time.time() - start_time))
        
        # test the model
        n_batches = int(MNIST.test.num_examples/model.BATCH_SIZE)
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = MNIST.test.next_batch(model.BATCH_SIZE)
            _, loss_batch, logits_batch = sess.run([model.graph['optimizer'], model.graph['loss'], model.graph['softmax']], 
                                            feed_dict={model.X: X_batch, model.Y:Y_batch, model.dropout: 1.0}) 
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)   
        
        print("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))

if __name__ == '__main__':
    
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    SKIP_STEP = 10
    DROPOUT = 0.75
    N_EPOCHS = 30
    N_CLASSES = 10
    
    MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)
    
    model = CNN(LEARNING_RATE,BATCH_SIZE,DROPOUT,N_EPOCHS,N_CLASSES)
    
    model._BuildGraph()
    
    train(MNIST, model)
    
tf.reset_default_graph()