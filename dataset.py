import numpy as np
import os
import cv2 as cv
import random
import time
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
    return imgs_coord

def get_image(imgtuple,size):

    img = cv.imread(imgtuple[0])
    x,y = imgtuple[1]
    img = img[x*size:(x+1)*size,y*size:(y+1)*size]
    return img

def load_imgs(data_dir,crop_size,shrunk_size):
    imgs_coord = get_files(data_dir,crop_size)
    input = []
    target = []
    for imgtuple in imgs_coord:
        img = get_image(imgtuple,crop_size)
        img_shrun = cv.resize(img,(shrunk_size,shrunk_size))
        input += [img_shrun]
        target += [img]
    input_seq = np.asarray(input)
    target_seq = np.asarray(target)
    return input,target

def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch