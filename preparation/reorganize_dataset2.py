"""
goal: image are all in dir: images, (root is "custom" which is not needed here
annotations are all in dir: labels
label names in file: classes.names
train and val sets are distinguished by file train.txt and file val.txt
"""
import torch
import os
import numpy as np
from shutil import copyfile
root = '/mnt/HDD1/SARAS'
dataset_root = '/mnt/HDD1/SARAS/dataset'
#
# files_val = os.listdir(os.path.join(root, 'val/obj'))
# files_set1 = os.listdir(os.path.join(root, 'train/set1'))
# files_set2 = os.listdir(os.path.join(root, 'train/set2'))

f_val = [os.path.join(root, 'val/obj', f) for f in os.listdir(os.path.join(root, 'val/obj'))]
f_set1 = [os.path.join(root, 'train/set1', f) for f in os.listdir(os.path.join(root, 'train/set1'))]
f_set2 = [os.path.join(root, 'train/set2', f) for f in os.listdir(os.path.join(root, 'train/set2'))]
f_train = f_set1 + f_set2
print(f_val[0].split('/')[-1])





print(os.path.getsize(os.path.join(root, 'val/obj', 'RARP1_frame_1.txt')))
print(os.stat(os.path.join(root, 'val/obj', 'RARP1_frame_1.txt')).st_size)

print(f_val[0].split('/')[-1].split('.')[0])

# print(len(os.listdir(os.path.join(dataset_root, 'images'))))
# print(len(os.listdir(os.path.join(dataset_root, 'labels'))))
#
with open(os.path.join(dataset_root, 'train.txt'), 'a') as file:
    for line in f_train:
        if line.split('.')[-1] == 'txt':  # label file
            if os.stat(line).st_size != 0:  # non-empty label file
                fn = line.split('/')[-1].split('.')[0] + '.jpg'
                file.write('data/custom/images/'+fn+'\n')

with open(os.path.join(dataset_root, 'valid.txt'), 'a') as file:
    for line in f_val:
        if line.split('.')[-1] == 'txt':  # label file
            if os.stat(line).st_size != 0:  # non-empty label file
                fn = line.split('/')[-1].split('.')[0] + '.jpg'
                file.write('data/custom/images/' + fn + '\n')