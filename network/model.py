from network.ops import *
import tensorflow as tf
from utils.compute import *
mean_x = 127  # tf.reduce_mean(self.input)
mean_y = 127  # tf.reduce_mean(self.input)

def net(input,reuse = False,args=None,name='EDSR'):
    with tf.variable_scope(name,reuse=reuse):
        L1 = conv_relu(input,args.EDFILTER_DIM,name='Conv2d_1')
        x = L1
        for i in range(args.nubBlocks_ED):
            x = resBlock_ED(x,args.EDFILTER_DIM,scale=args.resScale,name='Block_'+str(i))
        L2 = conv_relu(x,args.EDFILTER_DIM,name='conv2d_2')
        L_res = L2+L1
        L_U = upsample(L_res,args.EDFILTER_DIM,scale=args.scale)
        output = tf.clip_by_value(L_U+mean_x,0.0,255.0)
    return output

def RDN(input,reuse=False,is_training=True,args=None,name='RDN'):
    with tf.variable_scope(name,reuse=reuse):
        F_1 = conv_b(input,64,name='conv2d_1')
        F_0 = conv_b(F_1,64,name='conv2d_2')
        rdb_list = []
        rdb_in = F_0
        for i in range(1,17):
            x = rdb_in
            for j in range(1,9):
                tmp = conv_relu(x,64,name='RDB_'+str(i)+'_'+str(j))
                x = tf.concat([x,tmp],axis=3)
            x = conv_b(x,64,k_h=1,k_w=1,name='RDB_'+str(i))
            rdb_in = x+rdb_in
            rdb_list.append(rdb_in)
        FD = tf.concat(rdb_list,axis=3)
        FGF1 = conv_b(FD,64,k_h=1,k_w=1,name='conv2d_F1')
        FGF2 = conv_b(FGF1,64,name='conv2d_F2')
        FDF = tf.add(F_1,FGF2)

        ### upsamlple
        upsamp1 = conv_relu(FDF,64,name='conv_up1')
        upsamp2 = conv_relu(upsamp1,32,name='conv_up2')
        up = conv_b(upsamp2,48,name='conv_up3')
        up = pixelShuffler(up,4)

        conv_out = conv_b(up,3,name='conv_out')
        return conv_out




#### generator
def generator(input,reuse=False,is_training=True,args=None,name='SRResnet'):
    with tf.variable_scope(name,reuse=reuse):
        with tf.variable_scope('input_stage'):
            L1 = PReLU(conv_b(input,args.SPFILTER_DIM,k_h=9,k_w=9,name='conv2d_1'),name='PReLU_1')
            x = L1
            for i in range(args.nubBlocks_SR):
                x = resBlock_SR(x,args.SPFILTER_DIM,is_training=is_training,name='Block_'+str(i))
            L2 = conv_bn(x,args.SPFILTER_DIM,is_train=is_training,name='con2d_2')
            L2 = L2 +L1
        with tf.variable_scope('subpixelconv_stage1'):
            assert args.scale == 4
            L3_U1 = conv_b(L2,args.SPFILTER_DIM,name='connv2d_U1')
            L3_U1 = pixelShuffler(L3_U1,2)
            L3_U1 = PReLU(L3_U1,name='PReLU_U1')
            L4_U2 = conv_b(L3_U1,args.SPFILTER_DIM,name='conv2d_U2')
            L4_U2 = pixelShuffler(L4_U2,2)
            L4_U2 = PReLU(L4_U2,name='PReLU_U2')
        with tf.variable_scope('ouput_stage'):
            output = conv_b(L4_U2,3,k_h=9,k_w=9,name='conv2d_out')
        return output
# discriminator
def discriminator(input,reuse=False,is_training=True,args=None,name='discriminator'):
    with tf.variable_scope(name,reuse=reuse):

        L1 = LReLU(conv_b(input,64,name='conv2d_1'),leak=0.2,name='LReLU_1')

        ### block
        L1_block = LReLU(conv_bn(L1,64,strides=[1,2,2,1],is_train=is_training,name='con2d_block_L1'),leak=0.2,name='LReLU_block1')
        L2_block = LReLU(conv_bn(L1_block,128,is_train=is_training,name='con2d_block_L2'),leak=0.2,name='LReLU_block2')
        L3_block = LReLU(conv_bn(L2_block,128,strides=[1,2,2,1],is_train=is_training,name='con2d_block_L3'),leak=0.2,name='LReLU_block3')
        L4_block = LReLU(conv_bn(L3_block,256,is_train=is_training,name='con2d_block_L4'),leak=0.2,name='LReLU_block4')
        L5_block = LReLU(conv_bn(L4_block,256,strides=[1,2,2,1],is_train=is_training,name='con2d_block_L5'),leak=0.2,name='LReLU_block5')
        L6_block = LReLU(conv_bn(L5_block,512,is_train=is_training,name='con2d_block_L6'),leak=0.2,name='LReLU_block6')
        L7_block = LReLU(conv_bn(L6_block,512,strides=[1,2,2,1],is_train=is_training,name='con2d_block_L7'),leak=0.2,name='LReLU_block7')

        ### Dense
        L1_Dense = LReLU(tf.layers.dense(tf.layers.flatten(L7_block),units=512),leak=0.2,name='LReLU_Dense1')
        L2_Dense = tf.nn.sigmoid(tf.layers.dense(L1_Dense,units=1),name='sigmoid_Dense2')
        return L2_Dense

def discr_loss(output,label,EPS):
    ## 1
    discrim_fake_loss = tf.log(1 - output + EPS)
    discrim_real_loss = tf.log(label + EPS)
    discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

    ## 2
    # posLoss = tf.reduce_mean(tf.square(label - tf.random_uniform(shape=[label.get_shape().as_list()[0], 1], minval=0.9, maxval=1.0)))
    # negLoss = tf.reduce_mean(tf.square(output - tf.random_uniform(shape=[output.get_shape().as_list()[0], 1], minval=0, maxval=0.2, dtype=tf.float32)))
    # loss = posLoss+negLoss
    return discrim_loss

def gen_loss(output,label,dis_out,EPS,perceptual_mode):
    loss_mse = tf.reduce_mean(tf.square(output-label))

    if  perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(output, perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(label, perceptual_mode, reuse=True, scope=scope)

    loss_vgg = tf.reduce_mean(tf.square(extracted_feature_gen-extracted_feature_target))
    # loss2 = tf.reduce_mean(tf.square(discrNeg-tf.random_uniform(shape=[discrNeg.get_shape().as_list()[0],1],minval=0.9,maxval=1.1,dtype=tf.float32)))
    adversarial_loss = tf.reduce_mean(-tf.log(dis_out + EPS))
    gen_loss = loss_mse + 1e-3 * adversarial_loss + 2e-6*loss_vgg
    return gen_loss

def VGG19_slim(input, type, reuse,scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        # target_layer = scope + 'vgg_19/conv5/conv5_4'
        target_layer = 'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        # target_layer = scope + 'vgg_19/conv2/conv2_2'
        target_layer = 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]

    return output
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