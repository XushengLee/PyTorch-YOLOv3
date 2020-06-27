"""
examine whether the filenames of train and val set are unique.
the answer is yes.
"""
import os

root = '/mnt/HDD1/SARAS'

files_val = os.listdir(os.path.join(root, 'val/obj'))
print('val size', len(files_val))

files_set1 = os.listdir(os.path.join(root, 'train/set1'))
files_set2 = os.listdir(os.path.join(root, 'train/set2'))

print(len(set(files_val +files_set1 + files_set2)))

print(len(files_set1), len(files_set2), len(files_set1)+len(files_set2))


d = set(files_set1 +files_set2)
files_train = list(d)
print('train size', len(files_train)//2)

# print(files_val)

files_val = set([file.split('.')[0] for file in files_val])
print(len(files_val))

files_train = set([file.split('.')[0] for file in files_train])
print(len(files_train))

print(len(files_val) + len(files_train))
