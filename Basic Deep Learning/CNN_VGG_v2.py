import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io

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
     
    def _Weights(self, model_weights, layerNumber, expected_layer_name):
        
        if layerNumber == 0:
            W = model_weights[0][layerNumber][0][0][2][0][0][:,:,1,:]
            W = W.reshape(W.shape[0],W.shape[1],1,W.shape[2])
        elif layerNumber == 37:
            W = model_weights[0][layerNumber][0][0][2][0][0][1,1,:,:]
            #W = W.reshape(W.shape[0],W.shape[1])
        else:   
            W = model_weights[0][layerNumber][0][0][2][0][0]
        b =  model_weights[0][layerNumber][0][0][2][0][1]
        layer_name = model_weights[0][layerNumber][0][0][0][0]
    
        assert layer_name == expected_layer_name
        
        return W, b.reshape(b.size)
    
    def _AddConv2d(self, prev_layer, model_weights, stride, layer_number, scope_name):
        
        W, b = self._Weights(model_weights, layer_number, scope_name)
        
        #needed since input is a numpy array
        W = tf.constant(W)
        b = tf.constant(b)
        
        with tf.variable_scope(scope_name) as scope:
            kernel = tf.Variable(W, name = 'kernel',trainable = True)
            biases = tf.Variable(b, name = 'biases', trainable = True)
            conv = tf.nn.conv2d(prev_layer, filter = kernel, strides = stride, padding = 'SAME')
            conv1 = tf.nn.relu(conv + biases, name = scope.name)
        
        return conv1
        
    def _AddAvgPool(self, prev_layer, ksize, strides, scope_name):
        
        with tf.variable_scope(scope_name) as scope:
            pool = tf.nn.max_pool(prev_layer, ksize = ksize, strides = strides,padding='SAME')
        
        return pool
    
    def _AddFullyConnected(self, prev_layer, input_feature, scope_name, n_hidden, model_weights, layer_number, dropout):
        
        W, b = self._Weights(model_weights, layer_number, scope_name)
        
        #needed since input is a numpy array
        W = tf.constant(W)
        b = tf.constant(b)
        
        W = tf.reshape(W,[input_feature,n_hidden])
            
        with tf.variable_scope(scope_name) as scope:

            w = tf.Variable(W, name = 'weights')
            b = tf.Variable(b, name = 'biases')
            
            if layer_number == 37:
                pool = tf.reshape(prev_layer, [-1, input_feature])
            else:
                pool = prev_layer
            fc = tf.nn.relu(tf.matmul(pool, w) + b, name='relu')
            
            if dropout:
              fc = tf.nn.dropout(fc, self.dropout, name='relu_dropout')
        
        return fc
    
    def _AddSoftmax(self, FullyConnectedLayer, n_hidden):
        
        with tf.variable_scope('softmax_linear') as scope:
            w = tf.get_variable('weights', [n_hidden, self.N_CLASSES], initializer=tf.truncated_normal_initializer())
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
    
    def _BuildGraph(self, model_weights):
        
        images = tf.reshape(self.X, shape=[-1, 28, 28, 1]) 
        
        self.graph = {}
        
        #Define architecture here:
        
        # convoltuion block 1
        self.graph['conv1_1']   = self._AddConv2d(images, model_weights, [1, 1, 1, 1], 0, 'conv1_1')
        self.graph['conv1_2']   = self._AddConv2d(self.graph['conv1_1'], model_weights, [1, 1, 1, 1], 2, 'conv1_2')
        
        self.graph['pool1']   = self._AddAvgPool(self.graph['conv1_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool1')
        
        # convolution block 2
        self.graph['conv2_1']   = self._AddConv2d(self.graph['pool1'], model_weights, [1, 1, 1, 1], 5, 'conv2_1')
        self.graph['conv2_2']   = self._AddConv2d(self.graph['conv2_1'], model_weights, [1, 1, 1, 1], 7, 'conv2_2')
        
        self.graph['pool2']   = self._AddAvgPool(self.graph['conv2_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool2')
        
        # convolution block 3
        self.graph['conv3_1']   = self._AddConv2d(self.graph['pool2'], model_weights, [1, 1, 1, 1], 10, 'conv3_1')
        self.graph['conv3_2']   = self._AddConv2d(self.graph['conv3_1'], model_weights, [1, 1, 1, 1], 12, 'conv3_2')
        self.graph['conv3_3']   = self._AddConv2d(self.graph['conv3_2'], model_weights, [1, 1, 1, 1], 14, 'conv3_3')
        self.graph['conv3_4']   = self._AddConv2d(self.graph['conv3_3'], model_weights, [1, 1, 1, 1], 16, 'conv3_4')
        
        self.graph['pool3']   = self._AddAvgPool(self.graph['conv3_4'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool3')
        
        # convolution block 4
        self.graph['conv4_1']   = self._AddConv2d(self.graph['pool3'], model_weights, [1, 1, 1, 1], 19, 'conv4_1')
        self.graph['conv4_2']   = self._AddConv2d(self.graph['conv4_1'], model_weights, [1, 1, 1, 1], 21, 'conv4_2')
        self.graph['conv4_3']   = self._AddConv2d(self.graph['conv4_2'], model_weights, [1, 1, 1, 1], 23, 'conv4_3')
        self.graph['conv4_4']   = self._AddConv2d(self.graph['conv4_3'], model_weights, [1, 1, 1, 1], 25, 'conv4_4')
        
        self.graph['pool4']   = self._AddAvgPool(self.graph['conv4_4'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool4')
            
        # convolution block 5
        self.graph['conv5_1']   = self._AddConv2d(self.graph['pool4'], model_weights, [1, 1, 1, 1], 28, 'conv5_1')
        self.graph['conv5_2']   = self._AddConv2d(self.graph['conv5_1'], model_weights, [1, 1, 1, 1], 30, 'conv5_2')
        self.graph['conv5_3']   = self._AddConv2d(self.graph['conv5_2'], model_weights, [1, 1, 1, 1], 32, 'conv5_3')
        self.graph['conv5_4']   = self._AddConv2d(self.graph['conv5_3'], model_weights, [1, 1, 1, 1], 34, 'conv5_4')
        
        self.graph['pool5']   = self._AddAvgPool(self.graph['conv5_4'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool5')
        
        #fully connected layers
        self.graph['fc6']      = self._AddFullyConnected(self.graph['pool5'], 512, 'fc6', 4096, model_weights, 37, droput=True)
        self.graph['fc7']      = self._AddFullyConnected(self.graph['fc6'], 4096, 'fc7', 4096, model_weights, 39, droput=True)
        self.graph['fc8']      = self._AddFullyConnected(self.graph['fc7'], 4096, 'fc8', 1000, model_weights, 41, droput=False)
        
        self.graph['softmax'] = self._AddSoftmax(self.graph['fc8'],1000)
        self.graph['entropy'], self.graph['loss'] = self._CreateLoss(self.Y, self.graph['softmax'])
        
        #create optimizer
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
    DROPOUT = 0.5
    N_EPOCHS = 1
    N_CLASSES = 10
    
    vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    
    MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)
    
    model = CNN(LEARNING_RATE,BATCH_SIZE,DROPOUT,N_EPOCHS,N_CLASSES)
    
    model._BuildGraph(vgg['layers'])
   
    train(MNIST, model)
    
tf.reset_default_graph()

var_names = []    
for var in tf.trainable_variables():    
    var_names.append(var.name)


vgg['layers'][0][41][0][0][2][0][0].shape