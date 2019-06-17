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
## discriminator
# def discriminator(input,args=None):
#     def

