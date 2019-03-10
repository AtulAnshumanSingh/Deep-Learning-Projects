import tensorflow as tf
import time
import numpy as np
import scipy.io
import os
from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib
import matplotlib.pyplot as plt

class FCN:
    
    def __init__(self, LEARNING_RATE, BATCH_SIZE, DROPOUT, N_EPOCHS, N_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT):
        
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.N_CLASSES = N_CLASSES
        self.DROPOUT = DROPOUT
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        
        self.X = tf.placeholder(tf.float32, name='data')
    
        #self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
     
    def _Weights(self, model_weights, layerNumber, expected_layer_name):
        
        W = model_weights[0][layerNumber][1]
        b =  model_weights[0][layerNumber+1][1]
        layer_name = model_weights[0][layerNumber][0][0][:-7]
    
        assert layer_name == expected_layer_name
        
        return W, b.reshape(b.size)
    
    def _AddConv2d(self, prev_layer, model_weights, stride, layer_number, scope_name):
        
        W, b = self._Weights(model_weights, layer_number, scope_name)
        
        with tf.variable_scope(scope_name) as scope:
            kernel = tf.constant(W, name = 'kernel')
            biases = tf.constant(b, name = 'biases')
            conv = tf.nn.conv2d(prev_layer, filter = kernel, strides = stride, padding = 'SAME')
            conv1 = tf.nn.relu(conv + biases, name = scope.name)
        
        return conv1
        
    def _AddAvgPool(self, prev_layer, ksize, strides, scope_name):
        
        with tf.variable_scope(scope_name) as scope:
            pool = tf.nn.max_pool(prev_layer, ksize = ksize, strides = strides,padding='SAME')
        
        return pool
    
    def _AddDropout(self, prev_layer):
        
        return tf.nn.dropout(prev_layer, self.DROPOUT, name='relu_dropout')
    
    def _AddUpScore(self, prev_layer, model_weights, strides, layer_number, scope_name):
        
        W, b = self._Weights(model_weights, layer_number, scope_name)
        
        with tf.variable_scope(scope_name) as scope:
            deconv = tf.nn.conv2d_transpose(prev_layer, filter = W, strides = strides, padding = 'SAME',output_shape = [1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 21])
        
        return deconv
    
    def _CreateOptimizer(self,loss):
        
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss, global_step=self.global_step)
        
        return optimizer
        
    def _BuildGraph(self, model_weights):
        
        images = self.X #tf.reshape(self.X, shape=[-1, 500, 500, 3])
        
        self.graph = {}
        
        # define architecture here:
        
        # convoltuion block 1
        self.graph['conv1_1'] = self._AddConv2d(images, model_weights, [1, 1, 1, 1], 0, 'conv1_1')
        self.graph['conv1_2'] = self._AddConv2d(self.graph['conv1_1'], model_weights, [1, 1, 1, 1], 2, 'conv1_2')
        
        self.graph['pool1'] = self._AddAvgPool(self.graph['conv1_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool1')
        
        # convolution block 2
        self.graph['conv2_1'] = self._AddConv2d(self.graph['pool1'], model_weights, [1, 1, 1, 1], 4, 'conv2_1')
        self.graph['conv2_2'] = self._AddConv2d(self.graph['conv2_1'], model_weights, [1, 1, 1, 1], 6, 'conv2_2')
        
        self.graph['pool2'] = self._AddAvgPool(self.graph['conv2_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool2')
        
        # convolution block 3
        self.graph['conv3_1'] = self._AddConv2d(self.graph['pool2'], model_weights, [1, 1, 1, 1], 8, 'conv3_1')
        self.graph['conv3_2'] = self._AddConv2d(self.graph['conv3_1'], model_weights, [1, 1, 1, 1], 10, 'conv3_2')
        self.graph['conv3_3'] = self._AddConv2d(self.graph['conv3_2'], model_weights, [1, 1, 1, 1], 12, 'conv3_3')
        
        self.graph['pool3'] = self._AddAvgPool(self.graph['conv3_3'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool3')
        
        # convolution block 4
        self.graph['conv4_1'] = self._AddConv2d(self.graph['pool3'], model_weights, [1, 1, 1, 1], 14, 'conv4_1')
        self.graph['conv4_2'] = self._AddConv2d(self.graph['conv4_1'], model_weights, [1, 1, 1, 1], 16, 'conv4_2')
        self.graph['conv4_3'] = self._AddConv2d(self.graph['conv4_2'], model_weights, [1, 1, 1, 1], 18, 'conv4_3')
        
        self.graph['pool4'] = self._AddAvgPool(self.graph['conv4_3'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool4')
            
        # convolution block 5
        self.graph['conv5_1']  = self._AddConv2d(self.graph['pool4'], model_weights, [1, 1, 1, 1], 20, 'conv5_1')
        self.graph['conv5_2']  = self._AddConv2d(self.graph['conv5_1'], model_weights, [1, 1, 1, 1], 22, 'conv5_2')
        self.graph['conv5_3']  = self._AddConv2d(self.graph['conv5_2'], model_weights, [1, 1, 1, 1], 24, 'conv5_3')
        
        self.graph['pool5']  = self._AddAvgPool(self.graph['conv5_3'], [1, 2, 2, 1], [1, 2, 2, 1], 'pool5')
        
        # add fully conv layers
        self.graph['fc6'] = self._AddConv2d(self.graph['pool5'], model_weights, [1, 1, 1, 1], 26, 'fc6')
        self.graph['fc6'] = self._AddDropout(self.graph['fc6'])
        
        
        self.graph['fc7'] = self._AddConv2d(self.graph['fc6'], model_weights, [1, 1, 1, 1], 28, 'fc7')
        self.graph['fc7'] = self._AddDropout(self.graph['fc7'])  #40
        
        # one 1x1 conv layer
        self.graph['score_fr'] = self._AddConv2d(self.graph['fc7'], model_weights, [1, 1, 1, 1], 30, 'score_fr')
        
        # conv-transpose
        
        self.graph['upsample'] = self._AddUpScore(self.graph['score_fr'], model_weights, [1, 32, 32, 1], 32, 'upsample')
        
        # pred
        
        self.graph['pred_up'] = tf.argmax(self.graph['upsample'], dimension=3)
        
        # create summary
        #self.graph['summary_op'] = self._create_summary()
        
def get_resized_image(img_path, height, width, save=True):
    
    image = Image.open(img_path)
    '''image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)'''
    image = np.asarray(image, np.float32)
    
    return np.expand_dims(image, 0)

def color_seg(seg, palette):
    """
    Replace classes with their colors.
    Takes:
        seg: H x W segmentation image of class IDs
    Gives:
        H x W x 3 image of class colors
    """
    return palette[seg.flat].reshape(seg.shape + (3,))

def vis_seg(img, seg, palette, alpha=0.5):
    """
    Visualize segmentation as an overlay on the image.
    Takes:
        img: H x W x 3 image in [0, 255]
        seg: H x W segmentation image of class IDs
        palette: K x 3 colormap for all classes
        alpha: opacity of the segmentation in [0, 1]
    Gives:
        H x W x 3 image with overlaid segmentation
    """
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)
    return vis

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette
        
def color_image(image, num_classes=20):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))
    
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.5
N_EPOCHS = 1
N_CLASSES = 10

from six.moves import urllib

FCN_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/pascal-fcn32s-dag.mat'
FCN_MODEL = 'pascal-fcn32s-dag.mat'
    
urllib.request.urlretrieve(FCN_DOWNLOAD_LINK, FCN_MODEL)

ImageLink = 'https://thenypost.files.wordpress.com/2018/10/102318-dogs-color-determine-disesases-life.jpg?quality=90&strip=all&w=618&h=410&crop=1'
Imling = 'dogs1.jpg'
    
urllib.request.urlretrieve(ImageLink, Imling)

image = get_resized_image(Imling, 500, 500)

vgg16 = scipy.io.loadmat('pascal-fcn32s-dag.mat')

model = FCN(LEARNING_RATE,BATCH_SIZE,DROPOUT,N_EPOCHS,N_CLASSES, image.shape[1], image.shape[2])

model._BuildGraph(vgg16['params'])

sess = tf.Session()

final = sess.run([model.graph['pred_up']], feed_dict={model.X : image})

final = final[0]


'''color = color_image(final[0])
imageInput = Image.open(Imling)
plt.imshow(imageInput)
plt.imshow(color[0], cmap='jet', alpha=0.5)'''

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
in_ = np.array(image, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# visualize segmentation in PASCAL VOC colors
voc_palette = make_palette(21)
out_im = Image.fromarray(color_seg(final, voc_palette))
out_im.save('demo/output.png')
masked_im = Image.fromarray(vis_seg(image, final, voc_palette))
masked_im.save('demo/visualization.jpg')