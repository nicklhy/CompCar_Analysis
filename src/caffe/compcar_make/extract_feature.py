#!/usr/bin/env python

import os, sys
import time
import argparse
import cPickle
import numpy as np

CAFFE_ROOT = '/home/lhy/Documents/Codes/Libs/caffe'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe


parser = argparse.ArgumentParser(description='Train a xgbt classifier with CNN features')
parser.add_argument('--gpu', dest='gpu_id',
                    help='GPU device id to use [0]',
                    default=0, type=int)
parser.add_argument('--phase', dest='phase',
                    help='train or test images ?',
                    default='train', type=str)
parser.add_argument('--task', dest='task',
                    help='all front rear side front_side rear_side',
                    default='', type=str)
args = parser.parse_args()


task = ''
phase = 'train'
gpu_id = 0
if args.task is not None:
    task = args.task
if args.gpu_id is not None:
    gpu_id = args.gpu_id
if args.phase is not None:
    phase = args.phase

RANK_NUM = 1

DATA_ROOT = '/home/lhy/Documents/Data/CompCars'
im_list_file = '/home/lhy/Documents/Data/CompCars/train_test_split/classification/'+phase+task+'_make'+'.txt'
with open(im_list_file) as fd:
    gt = map(lambda s: s.strip().split(' '), fd.readlines())

FEAT_FILE = './cache/%s_deep_feat.pkl' % phase


if os.path.exists(FEAT_FILE):
    with open(FEAT_FILE, 'r') as fd:
        feat = cPickle.load(fd)
        print "feature file loaded from " + FEAT_FILE
else:
    net_module_file = './models/bvlc_googlenet/bvlc_googlenet_compcar_make_iter_200000.caffemodel'
    net_def_file = './models/bvlc_googlenet/deploy.prototxt'
    net = caffe.Net(net_def_file, net_module_file, caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    net.blobs['data'].reshape(1, 3, 224, 224)

    feats = np.zeros([len(gt), net.blobs['pool5/7x7_s1'].data.shape[1]])
    print 'feature extraction begins ... ...'
    for i, (img_path, label) in enumerate(gt):
        t1 = time.time()
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(os.path.join(DATA_ROOT, 'cropped_image', img_path)))
        out = net.forward()
        feats[i, :] = net.blobs['pool5/7x7_s1'].data.flatten()
        t2 = time.time()
        print img_path+' finished in %f seconds' % (t2-t1)
    print 'feature extraction finished ... ...'
    with open(FEAT_FILE, 'wb') as fd:
        cPickle.dump(feats, fd, cPickle.HIGHEST_PROTOCOL)
