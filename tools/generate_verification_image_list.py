import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

file_list = ['train_test_split/verification/verification_pairs_easy.txt',
             'train_test_split/verification/verification_pairs_medium.txt',
             'train_test_split/verification/verification_pairs_hard.txt']

img_list = set()
for fname in file_list:
    pairs = open(os.path.join(DATA_DIR, fname)).readlines()
    for pair in pairs:
        f1, f2, lb = pair.strip().split(' ')
        img_list.add(f1)
        img_list.add(f2)

with open(os.path.join(DATA_DIR, 'train_test_split', 'verification', 'verification_list.txt'), 'w') as fd:
    for img in img_list:
        fd.write('%s\n' % img)

