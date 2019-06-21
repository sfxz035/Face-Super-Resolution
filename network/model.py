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
            L3_U1 = conv_b(L2,args.SPFILTER_DIM*4,name='connv2d_U1')
            L3_U1 = pixelShuffler(L3_U1,2)
            L3_U1 = PReLU(L3_U1,name='PReLU_U1')
            L4_U2 = conv_b(L3_U1,args.SPFILTER_DIM*4,name='conv2d_U2')
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
        L1_Dense = LReLU(tf.layers.dense(tf.layers.flatten(L7_block),units=1024),leak=0.2,name='LReLU_Dense1')
        L2_Dense = tf.nn.sigmoid(tf.layers.dense(L1_Dense,units=1),leak=0.2,name='LReLU_Dense2')
    return L2_Dense

def discr_loss(output,label,EPS):
    ## 1
    discrim_fake_loss = tf.log(1 - output + EPS)
    discrim_real_loss = tf.log(label + EPS)
    discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))
    ## 2
    posLoss = tf.reduce_mean(tf.square(label - tf.random_uniform(shape=[label.get_shape().as_list()[0], 1], minval=0.9, maxval=1.0)))
    negLoss = tf.reduce_mean(tf.square(output - tf.random_uniform(shape=[output.get_shape().as_list()[0], 1], minval=0, maxval=0.2, dtype=tf.float32)))
    loss = posLoss+negLoss
def gen_loss(output,label,EPS):
    loss1 = tf.reduce_mean(tf.square(output-label))
