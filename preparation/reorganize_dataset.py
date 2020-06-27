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

files_val = os.listdir(os.path.join(root, 'val/obj'))
files_set1 = os.listdir(os.path.join(root, 'train/set1'))
files_set2 = os.listdir(os.path.join(root, 'train/set2'))
d_val = set(files_val)
d_set1 = set(files_set1)
d_set2 = set(files_set2)
d = set(files_set1 +files_set2)
files_train = list(d)
files_total = files_train + files_val
print(len(files_total))
# print(len(files_total))
# with open(dataset_root + '/gotest.txt', 'a') as txt_file:
#     for _ in range(5):
#         txt_file.write('hello\n')

# for file in files_total:
#     if file in d:  # from train set
#         pass
#     else:

# for file in files_total:
#     where = None
#     if file in d_val:
#         where = 'val/obj'
#     elif file in d_set1:
#         where = 'train/set1'
#     else:
#         where = 'train/set2'
#     if file.split('.')[1] == 'txt':
#         copyfile(os.path.join(root, where, file), os.path.join(dataset_root, 'labels', file))
#     else:
#         copyfile(os.path.join(root, where, file), os.path.join(dataset_root, 'images', file))

a = np.loadtxt(os.path.join(root, 'val/obj', 'RARP1_frame_1.txt'))
if a.size:
    print(a)
print(os.path.getsize(os.path.join(root, 'val/obj', 'RARP1_frame_1.txt')))

# print(len(os.listdir(os.path.join(dataset_root, 'images'))))
# print(len(os.listdir(os.path.join(dataset_root, 'labels'))))
#
# with open(os.path.join(dataset_root, 'train.txt'), 'a') as file:
#     for line in files_train:
#         if line.split('.')[1] != 'txt':
#             file.write('data/custom/images/'+line+'\n')
#
# with open(os.path.join(dataset_root, 'valid.txt'), 'a') as file:
#     for line in files_val:
#         if line.split('.')[1] != 'txt':
#             file.write('data/custom/images/'+line+'\n')