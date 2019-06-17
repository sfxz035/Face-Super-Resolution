import tensorflow as tf
import dataset
import network.model as model
import numpy as np
import os
import argparse

from utils.compute import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)

# epoch = 2000000
# batch_size = 10
# learning_rate = 0.0001
# a = 0.1
# savenet_path = './libSaveNet/savenet/'
# train_file = './data/train_HR'
# test_file = './data/test_HR'
# crop_size = 96
# scale = 2
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",default="./data/train_HR")
parser.add_argument("--test_file",default="./data/test_HR")
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--resScale",default=0.1,type=float)
parser.add_argument("--nubBlocks_ED",default=32,type=int)
parser.add_argument("--nubBlocks_SR",default=16,type=int)
parser.add_argument("--EDFILTER_DIM",default=256,type=int)
parser.add_argument("--SPFILTER_DIM",default=64,type=int)
parser.add_argument("--batch_size",default=1,type=int)
parser.add_argument("--savenet_path",default='./libSaveNet/savenet/')
parser.add_argument("--epoch",default=2000000,type=int)
parser.add_argument("--learning_rate",default=0.0001,type=int)
parser.add_argument("--crop_size",default=96,type=int)
parser.add_argument("--shrunk_size",default=96//4,type=int)


args = parser.parse_args()


x_train,y_train = dataset.load_imgs(args.train_file,crop_size=args.crop_size,shrunk_size=args.shrunk_size)
x_test,y_test = dataset.load_imgs(args.test_file,crop_size=args.crop_size,shrunk_size=args.shrunk_size)

def train(args):
    x = tf.placeholder(tf.float32,shape = [args.batch_size,args.shrunk_size,args.shrunk_size, 3])
    y_ = tf.placeholder(tf.float32,shape = [args.batch_size,args.crop_size,args.crop_size,3])
    # is_training = tf.placeholder(tf.bool)
    # dropout_value = tf.placeholder(tf.float32)  # 参与节点的数目百分比
    y = model.generator(x,args=args)
    loss = tf.reduce_mean(tf.square(y - y_))
    PSNR = compute_psnr(y,y_)
    # summary_op = tf.summary.scalar('trainloss', loss)
    # summary_op2 = tf.summary.scalar('testloss', loss)

    tf.summary.scalar('trainloss', loss)
    tf.summary.scalar('PSNR', PSNR)
    summary_op = tf.summary.merge_all()

    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([batch_norm_updates_op]):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=8)

    train_writer = tf.summary.FileWriter('./my_graph/train', sess.graph)
    test_writer = tf.summary.FileWriter('./my_graph/test')
    tf.global_variables_initializer().run()
    # last_file = tf.train.latest_checkpoint(savenet_path)
    # if last_file:
    #     tf.logging.info('Restoring model from {}'.format(last_file))
        # saver.restore(sess, last_file)
    count, m = 0, 0
    for ep in range(args.epoch):
        batch_idxs = len(x_train) // args.batch_size
        for idx in range(batch_idxs):
            batch_input = x_train[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_labels = y_train[idx * args.batch_size: (idx + 1) * args.batch_size]
            # batch_input, batch_labels = dataset.random_batch(x_train,y_train,batch_size)
            sess.run(train_step, feed_dict={x: batch_input, y_: batch_labels})
            count += 1
            # print(count)
            if count % 1 == 0:
                m += 1
                # batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, batch_size)
                batch_input_test = x_test[0 : args.batch_size]
                batch_labels_test = y_test[0 : args.batch_size]
                loss1 = sess.run(loss, feed_dict={x: batch_input,y_: batch_labels})
                loss2 = sess.run(loss, feed_dict={x: batch_input_test, y_: batch_labels_test})
                PSNR_train = sess.run(PSNR, feed_dict={x: batch_input,y_: batch_labels})
                PSNR_test = sess.run(PSNR, feed_dict={x: batch_input_test, y_: batch_labels_test})
                print("Epoch: [%2d], step: [%2d], train_loss: [%.8f],PSNR_train:[%.8f]" \
                      % ((ep + 1), count, loss1,PSNR_train), "\t", 'test_loss:[%.8f],PSNR_test:[%.8f]' % (loss2,PSNR_test))
                train_writer.add_summary(sess.run(summary_op, feed_dict={x: batch_input, y_: batch_labels}), m)
                test_writer.add_summary(sess.run(summary_op, feed_dict={x: batch_input_test,
                                                                     y_: batch_labels_test}), m)
            if (count + 1) % 10000 == 0:
                saver.save(sess, os.path.join(args.savenet_path, 'conv_unet%d.ckpt-done' % (count)))

train(args)