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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#MAT_KEY = ["DE_time","FE_time","BA_time"]
MAT_KEY = ["DE_time"]
TRAIN_DATASET_DIR = "/home/jochen/test/train_data"
TEST_DATASET_DIR = "/home/jochen/test/test_data"
TRAIN_TFRECORDS_FILENAME = 'train.tfrecords'
TEST_TFRECORDS_FILENAME = 'test.tfrecords'

## Variable Set for training
IMG_SIZE = 64

## read_TRFecord function
## turn on : 1 / turn off : 0
CHECK = 1

## you can change the "change_value" function on your design.
IMAGE_LIST = []
LABEL_LIST = []

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

def read_matlab_file(filename,mat_key):
    data = scio.loadmat(filename)
    key = list(data.keys())
    for s in key:            
        if s.find(mat_key) > 0:
            return data[s]

def change_value_2(mvalue):
    scaler1 = MinMaxScaler()     #matrix1 -> 0~1
    scaler1.fit(mvalue)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    mvalue = scaler1.transform(mvalue)

    mvalue = np.asarray(mvalue)
    mvalue = mvalue * 255    #matrix1 -> 0~255
    print(mvalue)

    return np.uint8(mvalue)

def change_value_3(mvalue):
    #sigmoid = 1 / (1 + np.exp(-mvalue))
    new_value = mvalue * 250
    new_value = np.around(new_value,0)
    new_value = abs(new_value)
    new_value[new_value>255] = 255

    return np.uint8(new_value)

def change_value(mvalue,label):
    mx = max(mvalue)
    mn = min(mvalue)
    M = [(c-mn)/(mx-mn) for c in mvalue]
    M = [c * 255 for c in M]
    new_value = np.uint8(M)

    IMAGE_LIST.append(new_value)
    LABEL_LIST.append(label)


def change_value_1(mvalue):
    # -20 ~ 20 => 0 ~ 40 => 41 vs 42
    value = mvalue * int((IMG_SIZE)/2-2) + int((IMG_SIZE)/2-2)
    value = np.uint8(abs(np.around(value,0)))
    new_value = np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.uint8)

    ## pictures value
    for i in range(IMG_SIZE):
        new_value[value[i]][i] = 255
    new_value = new_value.flatten()

    return new_value

def write_to_TFRecord():
    TFWriter = tf.python_io.TFRecordWriter(TRAIN_TFRECORDS_FILENAME)
    TFWriter2 = tf.python_io.TFRecordWriter(TEST_TFRECORDS_FILENAME)

    count_list = [i for i in range(len(LABEL_LIST))]
    r.shuffle(count_list)

    for i in range(len(count_list)):
        value = IMAGE_LIST[count_list[i]]
        label = LABEL_LIST[count_list[i]]

        value = value.tostring()

        ftrs = tf.train.Features(
            feature={'Label':int64_feature(label),
                    'image_raw':bytes_feature(value)})

        example = tf.train.Example(features = ftrs)
        n = r.randint(1,10)
        if(n == 10):
            TFWriter2.write(example.SerializeToString())
        else:
            TFWriter.write(example.SerializeToString())

    TFWriter.close()
    TFWriter2.close()


def read_key_value(s,file,labels):

    try:
        value = read_matlab_file(file,s)
        print("[key]:"+str(s))
        if value is None:
            show_message("Error image:"+file)
        else :
            label = int(labels)

            for j in range(int(len(value)/(IMG_SIZE * IMG_SIZE))):
                mvalue = (value[IMG_SIZE*IMG_SIZE*j:IMG_SIZE*IMG_SIZE*(j+1)]).flatten()
                change_value(mvalue,label)

    except IOError as e:
        show_message("Skip!")




def convert_to_TFRecords(values, labels, DATASET_DIR):
    n_samples = len(labels)    

    show_message("Transform start...")

    for i in np.arange(0,n_samples):
        for s in MAT_KEY:           
            read_key_value(s,values[i],labels[i])            
    
    write_to_TFRecord()
    show_message("Transform done!")

def read_TFRecords(TFRECORDS_FILENAME):
    record_iterator = tf.python_io.tf_record_iterator(path=TFRECORDS_FILENAME)
    name = TFRECORDS_FILENAME + ".txt"
    infile = open(name,"w")
    
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        image_string = (example.features.feature['image_raw'].bytes_list.value[0])
        label = (example.features.feature['Label'].int64_list.value[0])
        image_Id = np.fromstring(image_string, dtype=np.uint8)
        infile.write(str(label)+":")
        infile.write(str(image_Id)+"\n")

    infile.close()


def get_File_2(DATASET_DIR, n):
    images = []
    subfolders = []
    
    for dirPath, dirNames,fileNames in os.walk(DATASET_DIR):
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
    print("[subfolders]:"+str(subfolders))       
    image_list = list(subfolders[:,0])
    if n == 1:
        label_list = list(subfolders[:,1])
    else :
        label_list = [CORRECT_LABEL] * len(image_list)
    print("[label_list]:"+str(label_list))
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

def get_File(DATASET_DIR, n):
    # The images in each subfolder
    images = []
    # The subfolders
    subfolders = []

    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(DATASET_DIR):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))

        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))

    # To record the labels of the image dataset
    labels = []
    count = 0
    for a_folder in subfolders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count+=1

    subfolders = np.array([images, labels])
    #subfolders = subfolders.transpose()

    subfolders = subfolders[:, np.random.permutation(subfolders.shape[1])].T

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list

if __name__ == "__main__" :

    images, labels = get_File(TRAIN_DATASET_DIR, 1)
    infile = open('data.txt','w')
    for i in range(len(images)):
        infile.write(str(labels[i])+":"+str(images[i])+"\n")
    infile.close()
    convert_to_TFRecords(images,labels,TRAIN_DATASET_DIR)

    if(CHECK):
        read_TFRecords(TRAIN_TFRECORDS_FILENAME)
        read_TFRecords(TEST_TFRECORDS_FILENAME)
    