#!/usr/bin/env python

import os, sys
import time
import argparse
import cPickle
import numpy as np

CAFFE_ROOT = 'third-parties/caffe'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

parser = argparse.ArgumentParser(description='extract deep feature')
parser.add_argument('--list_file', dest='list_file',
                    help='image list file', type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='DATA ROOT PATH',
                    default='data/cropped_image', type=str)
parser.add_argument('--output', dest='output',
                    help='output ',
                    default='', type=str)
parser.add_argument('--model_def', dest='model_def',
                    help='network definition file',
                    default='models/bvlc_googlenet_deploy.prototxt', type=str)
parser.add_argument('--weights', dest='weights',
                    help='model weights file',
                    default='models/bvlc_googlenet.caffemodel', type=str)
parser.add_argument('--gpu_id', dest='gpu_id',
                    help='GPU device to use [0]', default=0, type=int)
args = parser.parse_args()

if args.list_file == '':
    print 'please input a image list file'
    sys.exit(-1)


DATA_ROOT = args.data_dir
im_list_file = args.list_file
with open(im_list_file) as fd:
    im_list = [os.path.join(DATA_ROOT, x) for x in map(lambda s: s.strip(), fd.readlines())]

if args.output=='':
    FEAT_FILE = './cache/%s_deep_feat.pkl' % args.list_file
else:
    FEAT_FILE = args.output


if os.path.exists(FEAT_FILE):
    with open(FEAT_FILE, 'r') as fd:
        feat = cPickle.load(fd)
        print "feature file loaded from " + FEAT_FILE
else:
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.model_def, args.weights, caffe.TEST)
    caffe.set_mode_gpu()
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    net.blobs['data'].reshape(1, 3, 224, 224)

    feats = np.zeros([len(im_list), net.blobs['pool5/7x7_s1'].data.shape[1]])
    print 'feature extraction begins ... ...'
    for i, img_path in enumerate(im_list):
        t1 = time.time()
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_path))
        out = net.forward()
        feats[i, :] = net.blobs['pool5/7x7_s1'].data.flatten()
        t2 = time.time()
        print img_path+' finished in %f seconds' % (t2-t1)
    print 'feature extraction finished ... ...'
    feats = (im_list, feats)
    with open(FEAT_FILE, 'wb') as fd:
        cPickle.dump(feats, fd, cPickle.HIGHEST_PROTOCOL)
