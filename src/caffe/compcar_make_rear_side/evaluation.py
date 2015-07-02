#!/usr/bin/env python

import os
import sys
import time
import cPickle
import numpy as np
import scipy.io as sio

CAFFE_ROOT = '/home/lhy/Documents/Codes/Libs/caffe'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

task = '_rear_side'
RANK_NUM = 1

if len(sys.argv) > 1:
    gpu_id = int(sys.argv[1])
else:
    gpu_id = 0
prototxt = './models/bvlc_googlenet/deploy.prototxt'
caffemodel = './models/bvlc_googlenet/bvlc_googlenet_compcar_make'+task+'_iter_200000.caffemodel'

if not os.path.exists('./data'):
    os.mkdir('data')

res_file = './data/res.pkl'
DATA_ROOT = '/home/lhy/Documents/Data/CompCars'
im_list_file = '/home/lhy/Documents/Data/CompCars/train_test_split/classification/test'+task+'_make'+'.txt'

CONF_THRESHOLD = 0.3

with open(im_list_file) as fd:
    gt = map(lambda s: s.strip().split(' '), fd.readlines())

make_model_name = sio.loadmat(os.path.join(DATA_ROOT, 'misc', 'make_model_name.mat'), squeeze_me = True)
makes = make_model_name['make_names'].tolist()
# models = make_model_name['model_names'].tolist()
CLASSES = tuple(makes)
CLASS_NUM = len(makes)


if os.path.exists(res_file):
    with open(res_file, 'rb') as fd:
        det_res = cPickle.load(fd)
        # print '\nLoaded det_files\n'
else:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Classifier(prototxt,
                           caffemodel,
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(224, 224))
    print '\nLoaded network {:s}'.format(caffemodel)

    det_res = []
    for img, label in gt:
        im = caffe.io.load_image(os.path.join(DATA_ROOT, 'cropped_image', img))
        t1 = time.time()
        pred = net.predict([im])[0]
        t2 = time.time()
        print 'finish processing %s in %f s' % (img, t2-t1)
        y = np.argsort(-pred)
        ids = pred[y] > CONF_THRESHOLD
        det_res.append([img, int(label), y[ids], pred[y][ids]])
    with open(res_file, 'wb') as fd:
        cPickle.dump(det_res, fd, cPickle.HIGHEST_PROTOCOL)
    print '\n\nwrote results to %s\n' % res_file

total_hit_num = 0
total_num = len(det_res)
hit_num = np.zeros([250, ])
num = np.zeros([250, ])
for img, label, y, scores in det_res:
    n_pred = y.shape[0]
    num[int(label)-1] += 1
    if np.any(y[:min(n_pred, RANK_NUM)] == int(label)):
        total_hit_num += 1
        hit_num[int(label)-1] += 1

print 'rank %d average recognition rate = %f(%d/%d)\n' % (RANK_NUM, 1.0*total_hit_num/total_num, total_hit_num, total_num)
