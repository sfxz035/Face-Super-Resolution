import numpy as np
import os
import cv2 as cv
import random
import scipy.misc
import time
batch_index = 0
def get_files(file_dir,crop_size):
    imgs_coord = []
    img_files = os.listdir(file_dir)
    for file in img_files:
        img_dir = os.path.join(file_dir,file)
        img = cv.imread(img_dir)
        x,y,z = img.shape
        coords_x = x // crop_size
        coords_y = y // crop_size
        coords = [(q, r) for q in range(coords_x) for r in range(coords_y)]
        for coord in coords:
            imgs_coord.append((img_dir,coord))
    random.shuffle(imgs_coord)
    nub = str(len(imgs_coord))
    print('data number is '+nub)
    return imgs_coord

def get_image(imgtuple,size):

    img = cv.imread(imgtuple[0])
    x,y = imgtuple[1]
    img = img[x*size:(x+1)*size,y*size:(y+1)*size]
    return img
def get_batch(train_set,batch_size,crop_size,shrunk_size):
    global batch_index
    input = []
    target = []
    max_counter = len(train_set)//batch_size
    counter = batch_index % max_counter
    window = [x for x in range(counter*batch_size,(counter+1)*batch_size)]
    for q in window:
        img = train_set[q]
        y = get_image(img,crop_size)
        x = scipy.misc.imresize(y,(shrunk_size,shrunk_size),'bicubic')
        input += [x]
        target += [y]
    input_seq = np.asarray(input)
    target_seq = np.asarray(target)

    batch_index = (batch_index+1)%max_counter
    return input_seq,target_seq
def load_imgs(data_dir,crop_size,shrunk_size,min=None):
    imgs_coord = get_files(data_dir,crop_size)
    if min != None:
        imgs_coord = imgs_coord[:min]
    input = []
    target = []
    i = 0
    for imgtuple in imgs_coord:
        img = get_image(imgtuple,crop_size)
        img_shrun = scipy.misc.imresize(img,(shrunk_size,shrunk_size),'bicubic')
        img = img/(255. / 2.)-1
        img_shrun = img_shrun/(255. / 2.)-1
        input += [img_shrun]
        target += [img]
        if i%100==0:
            print('data load...: '+str(i))
        i+=1
    input_seq = np.asarray(input)
    target_seq = np.asarray(target)
    # cv.namedWindow('a',0)
    # cv.imshow('a',input_seq[0][:])
    # cv.namedWindow('b',0)
    # cv.imshow('b',target_seq[0][:])
    # cv.waitKey(0)
    return input_seq,target_seq

def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch