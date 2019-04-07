#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:11:31 2019

@author: cat
"""

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import cv2
import random as r
import scipy.io as scio

MAT_KEY = "DE_time"
TRAIN_DATASET_DIR = "/home/cat/test/train_data"
TFRECORDS_FILENAME = 'test.tfrecords'

## Variable Set for training
IMG_SIZE = 42

## read_TRFecord function
## turn on : 1 / turn off : 0
CHECK = 1

## you can change the "change_value" function on your design.

def show_message(s):
    for i in range(len(s)+10):
        print("#",end='')
    print("\n#    "+str(s)+"    #")
    for j in range(len(s)+10):
        print("#",end='')
    print('\n')

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def float64_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def read_matlab_file(filename):
    data = scio.loadmat(filename)
    key = list(data.keys())
    for s in key:            
        if s.find(MAT_KEY) > 0:
            return data[s]

def change_value(mvalue):
    #sigmoid = 1 / (1 + np.exp(-mvalue))
    new_value = mvalue * 255
    new_value = np.around(new_value,0)

    return np.uint8(abs(new_value))

def convert_to_TFRecords(values, labels):
    n_samples = len(labels)
    TFWriter = tf.python_io.TFRecordWriter(TFRECORDS_FILENAME)

    show_message("Transform start...")

    for i in np.arange(0,n_samples):
        try:
            value = read_matlab_file(values[i])
            if value is None:
                show_message("Error image:"+values[i])
            label = int(labels[i])

            for j in range(int(len(value)/(IMG_SIZE*IMG_SIZE))):
                mvalue = (value[IMG_SIZE*IMG_SIZE*j:IMG_SIZE*IMG_SIZE*(j+1)]).flatten()
                mvalue = change_value(mvalue)
                mvalue = mvalue.tostring()

                ftrs = tf.train.Features(
                    feature={'Label':int64_feature(label),
                            'image_raw':bytes_feature(mvalue)})

                example = tf.train.Example(features = ftrs)
                TFWriter.write(example.SerializeToString())

        except IOError as e:
            show_message("Skip!")
            
    TFWriter.close()
    show_message("Transform done!")

def read_TFRecords():
    record_iterator = tf.python_io.tf_record_iterator(path=TFRECORDS_FILENAME)
    infile = open("detail.txt","w")
    
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        image_string = (example.features.feature['image_raw'].bytes_list.value[0])
        label = (example.features.feature['Label'].int64_list.value[0])
        image_Id = np.fromstring(image_string, dtype=np.uint8)
        infile.write(str(label)+":")
        infile.write(str(image_Id)+"\n")

    infile.close()


def get_File():
    images = []
    subfolders = []
    
    for dirPath, dirNames,fileNames in os.walk(TRAIN_DATASET_DIR):
        names = []
        for name in fileNames:
            names.append(os.path.join(dirPath, name))
        for name in dirNames:
            subfolders.append(os.path.join(dirPath,name))

        r.shuffle(names)
        if names != []:
            images.append(names)

    mincount = float("Inf")

    for num_folder in subfolders:
        n_img = len(os.listdir(num_folder))
        if n_img < mincount:
            mincount = n_img

    # Keep only the minimum number of files
    for i in range(len(images)):
        images[i] = images[i][0:mincount]

    images = np.reshape(images,[mincount*len(subfolders),])
    labels = []
    for count in range(len(subfolders)):
        labels = np.append(labels, mincount * [count])

    # Disrupt the order of the final output,
    # removing the gap between each category
    subfolders = np.array([images,labels])
    subfolders = subfolders[:,np.random.permutation(subfolders.shape[1])].T
        
    image_list = list(subfolders[:,0])
    label_list = list(subfolders[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

if __name__ == "__main__" :
    images, labels = get_File()
    convert_to_TFRecords(images,labels)
    if(CHECK):
        read_TFRecords()
    