# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim


def weight_variable(shape,name=None,trainable=True, decay_mult = 0.0):
    weights = tf.get_variable(
        name, shape, tf.float32, trainable=trainable,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
        # initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
        # regularizer=tf.contrib.layers.l2_regularizer(decay_mult)
    )
    return weights

def bias_variable(shape,name=None, bias_start = 0.0, trainable = True, decay_mult = 0.0):
    bais = tf.get_variable(
        name, shape, tf.float32, trainable = trainable,
        initializer = tf.constant_initializer(bias_start, dtype = tf.float32)
        # regularizer = tf.contrib.layers.l2_regularizer(decay_mult)
    )
    return bais

def conv_bn(inpt ,output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1], is_train = True, name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h,k_w,inpt.get_shape()[-1],output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        batch_norm = tf.layers.batch_normalization(conv, training=is_train) ###由contrib换成layers
    return batch_norm

def BatchNorm(
        value, is_train = True, name = 'BatchNorm',
        epsilon = 1e-5, momentum = 0.9
    ):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(
            value,
            decay = momentum,
            # updates_collections = tf.GraphKeys.UPDATE_OPS,
            # updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = is_train,
            scope = name
        )





def conv_b(inpt, output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1],name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h, k_w, inpt.get_shape()[-1], output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        biases = bias_variable(output_dim,name='biases')
        out = tf.nn.bias_add(conv, biases)
    return out
def conv_relu(inpt, output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1],name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h, k_w, inpt.get_shape()[-1], output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        biases = bias_variable(output_dim,name='biases')
        pre_relu = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(pre_relu)
        return out





def SeLU(value, name = 'SeLU'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(value >= 0.0, value, alpha * tf.nn.elu(value))


def ELU(value, name = 'ELU'):
    with tf.variable_scope(name):
        return tf.nn.elu(value)

def ReLU(value, name = 'ReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)
def PReLU(_x,name='PReLU'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.25), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

def LReLU(x, leak = 0.333, name = 'LReLU'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name = name)

def Deconv2d(
        value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
        name = 'Deconv2d', with_w = False
    ):
    with tf.variable_scope(name):
        weights = weight_variable(
            name='weights',
            shape=[k_h, k_w, output_shape[-1], value.get_shape()[-1]],
            decay_mult = 1.0
        )
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides = strides
        )
        biases = bias_variable(name='biases', shape=[output_shape[-1]])
        deconv = tf.nn.bias_add(deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv
def Deconv2d_bn(
        value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
        is_train=True, name = 'Deconv2d', with_w = False
    ):
    with tf.variable_scope(name):
        weights = weight_variable(
            name='weights',
            shape=[k_h, k_w, output_shape[-1], value.get_shape()[-1]],
            decay_mult = 1.0
        )
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides = strides
        )
        batch_norm = tf.layers.batch_normalization(deconv, training=is_train) ###由contrib换成layers
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return batch_norm, weights
        else:
            return batch_norm

def resBlock_ED(input,feature_size=64,kernel_size=[3,3],strides=[1,1,1,1],scale=1,name='resBlock'):
    with tf.variable_scope(name):
        conv1 = conv_relu(input,feature_size,kernel_size[0],kernel_size[1],strides=strides,name='conv1')
        conv2 = conv_b(conv1,feature_size,kernel_size[0],kernel_size[1],strides=strides,name='conv2')
        out = conv2*scale+input
        return out
def resBlock_SR(input,feature_size=64,kernel_size=[3,3],strides=[1,1,1,1],is_training=True,name='resBlock'):
    with tf.variable_scope(name):
        conv1 = PReLU(conv_bn(input,feature_size,kernel_size[0],kernel_size[1],strides=strides,is_train=is_training,name='Resblock_conv1'),name='Resblock_PReLLU1')
        conv2 = conv_bn(conv1,feature_size,kernel_size[0],kernel_size[1],strides=strides,is_train=is_training,name='Resblock_conv2')
        output = conv2 + input
        return output



#### unsample
def upsample(x,features=64,scale=2,kernel_size=[3,3],strides=[1,1,1,1]):
    assert scale in [2,3,4]
    x = conv_b(x,features,kernel_size[0],kernel_size[1],strides=strides,name='upconv1')
    ps_features = 3*(scale**2)
    x = conv_b(x, ps_features, kernel_size[0], kernel_size[1], strides=strides,name='upconv2')
    x = PS(x,scale,color=True)
    return x

### 直接变换成3通道图片输出
"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""
def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc],3)
    else:
        X = _phase_shift(X, r)
    return X

####  根据目前通道数进行尺寸变换，不一定为3通道图片输出。可后接conv实现3通道输出。
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

### vgg
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points