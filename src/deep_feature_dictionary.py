#!/usr/bin/env python

import argparse
import os, sys
import time
import cPickle
import spams

parser = argparse.ArgumentParser(description='dictionary learning')
parser.add_argument('--feature_file', dest='feature_file',
                    help='feature file',
                    default='cache/train_deep_feat.pkl', type=str)
args = parser.parse_args()

if not os.path.exists(args.feature_file):
    print 'feature file does not exist!'
    sys.exit(-1)

with open(args.feature_file, 'r') as fd:
    t1 = time.time()
    feat = cPickle.load(fd)
    t2 = time.time()
    feat = feat[1].T
    print "feature file loaded from " + args.feature_file + ' in %f seconds' % (t2-t1)

dl_params = {'K': 100,
             'lambda1' : 0.15,
             'numThreads' : 4,
             'batchsize' : 400,
             'iter': 1000}

t1 = time.time()
D = spams.trainDL(feat, **dl_params)
t2 = time.time()
print 'Dictionary Learning: finish %d iterations in %f seconds' % (dl_params['iter'], t2-t1)
