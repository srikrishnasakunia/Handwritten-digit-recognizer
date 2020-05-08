# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:31:45 2020

@author: Krishna
"""
#%%
import os
import numpy as np
import glob

from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam, RMSprop


#%%
d = {}
from subprocess import check_output
#print(check_output(["ls","C:\Users\Krishna\Desktop\AI Project\forms_for_parsing.txt"]).decode("utf8"))
#forms = pd.read_csv('C:\Users\Krishna\Desktop\AI Project\forms_for_parsing.txt',header = None)
#print(forms.head)
with open('C:\\Users\\Krishna\\Desktop\\AI Project\\forms_for_parsing.txt') as f:
    for line in f:
        key = line.split(' ')[0]
        writer = line.split(' ')[1]
        d[key] = writer
print(len(d.keys()))  


#%%
tmp = []
target_list = []

path_to_files = os.path.join('C:\\Users\\Krishna\\Desktop\\AI Project\\data_subset\\data_subset', '*')
for filename in sorted(glob.glob(path_to_files)):
 #   print(filename)
    tmp.append(filename)
    image_name = filename.split('/')[-1]
    file, ext = os.path.splitext(image_name)
    parts = file.split('-')
    form = parts[0] + '-' + parts[1]
    #print(form)
    for key in d:
        if key == form:
            target_list.append(str(d[key]))

img_files = np.asarray(tmp)
img_targets = (np.asarray(target_list)).astype(str)
print(img_files.shape)
print(img_targets.shape)
        