#!/usr/bin/env python

import os
import sys
import cv2
from multiprocessing import Pool
import time


DATA_ROOT = './data'


if not os.path.exists(os.path.join(DATA_ROOT, 'cropped_image')):
    os.mkdir(os.path.join(DATA_ROOT, 'cropped_image'))


def fun(img_path):
    assert(os.path.exists(os.path.join(DATA_ROOT, 'image', img_path)) and
           os.path.exists(os.path.join(DATA_ROOT, 'label', img_path.replace('jpg', 'txt'))))

    lb_fd = open(os.path.join(DATA_ROOT, 'label', img_path.replace('jpg', 'txt')))
    label = lb_fd.readlines()
    x1, y1, x2, y2 = map(int, label[-1].split(' '))
    lb_fd.close()
    im = cv2.imread(os.path.join(DATA_ROOT, 'image', img_path))
    cropped_im = im[y1:y2, x1:x2, :]

    make, model, year, filename = img_path.replace('jpg', 'txt').split('/')
    if not os.path.exists(os.path.join(DATA_ROOT, 'cropped_image', make, model, year)):
        try:
            os.makedirs(os.path.join(DATA_ROOT, 'cropped_image', make, model, year))
        except:
            pass
    cv2.imwrite(os.path.join(os.path.join(DATA_ROOT, 'cropped_image', img_path)), cropped_im)


if len(sys.argv)>1:
    process_num = int(sys.argv[1])
else:
    process_num = 2
for f_name in ['./data/train_test_split/classification/train.txt',
               './data/train_test_split/classification/test.txt']:
    img_list = map(lambda s: s.strip(), open(f_name).readlines())
    print 'start processing %d images' % len(img_list)
    t1 = time.time()
    pool = Pool(process_num)
    pool.map(fun, img_list)
    t2 = time.time()
    print 'finish in %f s' % (t2-t1)
