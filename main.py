import tensorflow as tf
import dataset
import network.model as model
import numpy as np
import os
import argparse
import cv2 as cv
import scipy.misc

from utils.compute import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)


parser = argparse.ArgumentParser()
parser.add_argument("--train_file",default="./data/train_HR")
parser.add_argument("--test_file",default="./data/test_HR")
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--resScale",default=0.1,type=float)
parser.add_argument("--nubBlocks_ED",default=32,type=int)
parser.add_argument("--nubBlocks_SR",default=16,type=int)
parser.add_argument("--EDFILTER_DIM",default=256,type=int)
parser.add_argument("--SPFILTER_DIM",default=64,type=int)
parser.add_argument("--batch_size",default=10,type=int)
parser.add_argument("--savenet_path",default='./libSaveNet/savenet/')
parser.add_argument("--epoch",default=200000,type=int)
parser.add_argument("--learning_rate",default=0.0001,type=float)
parser.add_argument("--crop_size",default=384,type=int)
parser.add_argument("--shrunk_size",default=386//4,type=int)
parser.add_argument("--num_train",default=10000,type=int)
parser.add_argument("--num_test",default=1500,type=int)
parser.add_argument("--EPS",default=1e-12,type=float)



args = parser.parse_args()

def train(args):
    # train_set = dataset.get_files(args.train_file,crop_size=args.crop_size)
    # test_set = dataset.get_files(args.test_file,crop_size=args.crop_size)
    x_train, y_train = dataset.load_imgs(args.train_file, crop_size=args.crop_size, shrunk_size=args.shrunk_size,min=args.num_train)
    x_test, y_test = dataset.load_imgs(args.test_file, crop_size=args.crop_size, shrunk_size=args.shrunk_size,min=args.num_test)

    x = tf.placeholder(tf.float32,shape = [args.batch_size,args.shrunk_size,args.shrunk_size, 3])
    y_ = tf.placeholder(tf.float32,shape = [args.batch_size,args.crop_size,args.crop_size,3])
    y = model.generator(x,args=args)
    loss = tf.reduce_mean(tf.square(y - y_))


    PSNR = compute_psnr(y,y_,convert=True)

    tf.summary.scalar('trainloss', loss)
    tf.summary.scalar('PSNR', PSNR)
    summary_op = tf.summary.merge_all()

    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([batch_norm_updates_op]):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=20)

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
            # batch_input,batch_labels = dataset.get_batch(train_set,args.batch_size,args.crop_size,args.shrunk_size)
            batch_input = x_train[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_labels = y_train[idx * args.batch_size: (idx + 1) * args.batch_size]
            # batch_input, batch_labels = dataset.random_batch(x_train,y_train,batch_size)

            sess.run(train_step, feed_dict={x: batch_input, y_: batch_labels})
            count += 1
            # print(count)
            if count % 100 == 0:
                m += 1
                # batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, args.batch_size)
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
                saver.save(sess, os.path.join(args.savenet_path, 'conv_net%d.ckpt-done' % (count)))
def GAN_train(args):
    x_train, y_train = dataset.load_imgs(args.train_file, crop_size=args.crop_size, shrunk_size=args.shrunk_size)
    x_test, y_test = dataset.load_imgs(args.test_file, crop_size=args.crop_size, shrunk_size=args.shrunk_size)

    genInput = tf.placeholder(tf.float32,shape = [args.batch_size,args.shrunk_size,args.shrunk_size, 3])
    genLabel = tf.placeholder(tf.float32,shape = [args.batch_size,args.crop_size,args.crop_size,3])
    genOutput = model.generator(genInput,args=args,name='generator')

    discr_outlabel = model.discriminator(genLabel,name='discriminator')
    discr_outGenout = model.discriminator(genOutput,reuse=True,name='discriminator')

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list,max_to_keep=10)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    genvar_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disvar_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    modelsave = tf.train.Saver(var_list,max_to_keep=10)

    gen_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='generator'))
    # with tf.control_dependencies([gen_updates_op]):
    #     gentrain_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss,var_list=genvar_list)
    # dis_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='discriminator'))
    # with tf.control_dependencies([dis_updates_op]):
    # gentrain_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss,var_list=disvar_list)

    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)

    tf.global_variables_initializer().run()

    # vgg_restore.restore(sess, FLAGS.vgg_ckpt)
    # last_file = tf.train.latest_checkpoint(vgg_ckpt_path)
    # if last_file:
    #     saver.restore(sess, last_file)


def test(args):
    savepath = './libSaveNet/savenet/conv_net429999.ckpt-done'
    path_2k = './data/valid/0801.png'
    path_set = './data/valid/comic.png'
    path_face = './data/valid/2019-04-18-09-33-59-828886_1.bmp'
    img = cv.imread(path_set)
    img_shape = np.shape(img)
    a_scale = img_shape[0]//args.scale
    b_scale = img_shape[1]//args.scale
    img = img[0:a_scale*args.scale,0:b_scale*args.scale]
    img_LR = scipy.misc.imresize(img,(a_scale,b_scale),'bicubic')
    ## 归一化
    img_norm = img / (255. / 2.) - 1
    img_LR_norm = img_LR / (255. / 2.) - 1

    img_LR_input = np.expand_dims(img_LR_norm,0)
    img_label = np.expand_dims(img_norm,0)
    x = tf.placeholder(tf.float32,shape = [1,a_scale,b_scale, 3])
    y_ = tf.placeholder(tf.float32,shape = [1,a_scale*args.scale,b_scale*args.scale,3])
    y = model.generator(x,args=args,is_training=False)
    loss = tf.reduce_mean(tf.square(y - y_))
    PSNR = compute_psnr(y,y_,convert=True)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    output = sess.run(y,feed_dict={x:img_LR_input})

    loss_test = sess.run(loss,feed_dict={y:output,y_:img_label})
    PSNR_test = sess.run(PSNR,feed_dict={y:output,y_:img_label})


    np.save('./output/sp_img.npy',output)
    # cv.imwrite('./output/sp_img.png',output,0)
    # cv.imwrite('./output/lr_img.png',img_LR,0)
    # cv.imwrite('./output/hr_img.png',img,0)
    print('loss_test:[%.8f],PSNR_train:[%.8f]' % (loss_test,PSNR_test))
def predict(args):
    savepath = './libSaveNet/savenet/conv_net429999.ckpt-done'
    path_face = './data/valid/2019-04-18-09-33-59-828886_1.bmp'
    img = cv.imread(path_face)
    img = img / (255. / 2.) - 1
    img_shape = np.shape(img)
    img_input = np.expand_dims(img,0)
    x = tf.placeholder(tf.float32,shape = [1,img_shape[0],img_shape[1], 3])
    y = model.generator(x,args=args,is_training=False)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    output = sess.run(y,feed_dict={x:img_input})
    np.save('./output/sp_img.npy',output)

if __name__ == '__main__':
    # train(args)
    # test(args)
    predict(args)