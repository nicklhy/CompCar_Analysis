#!/usr/bin/env python

import os
import sys
import time
import argparse
import cPickle

XGBOOST_PATH = '/home/lhy/Documents/Codes/Libs/xgboost'
sys.path.append(os.path.join(XGBOOST_PATH, 'wrapper'))
import xgboost as xgb

parser = argparse.ArgumentParser(description='Train a xgbt classifier with CNN features')
parser.add_argument('--num_round', dest='num_round',
                    help='iteration rounds',
                    default=10, type=int)
args = parser.parse_args()
task = ''

TRAIN_FEAT_FILE = './cache/train_deep_feat.pkl'
TEST_FEAT_FILE = './cache/test_deep_feat.pkl'
param = {'objective': 'multi:softmax',
         'eta': 0.1,
         'max_depth': 6,
         'silent': 1,
         'nthread': 12,
         'num_class': 163}

num_round = args.num_round

model_path = './models/xgbt/%d.model' % num_round

DATA_ROOT = '/home/lhy/Documents/Data/CompCars'
train_list_file = '/home/lhy/Documents/Data/CompCars/train_test_split/classification/train'+task+'_make'+'.txt'
test_list_file = '/home/lhy/Documents/Data/CompCars/train_test_split/classification/test'+task+'_make'+'.txt'
with open(train_list_file) as fd:
    train_gt = map(lambda s: s.strip().split(' '), fd.readlines())
with open(test_list_file) as fd:
    test_gt = map(lambda s: s.strip().split(' '), fd.readlines())


if not (os.path.exists(TRAIN_FEAT_FILE) and
        os.path.exists(TEST_FEAT_FILE)):
    print 'feature files does not exists'
    sys.exit(-1)
else:
    with open(TRAIN_FEAT_FILE) as fd:
        train_feats = cPickle.load(fd)
    with open(TEST_FEAT_FILE) as fd:
        test_feats = cPickle.load(fd)
    print 'feature files loaded'
train_labels = [int(x[1]) for x in train_gt]
test_labels = [int(x[1]) for x in test_gt]

dtrain = xgb.DMatrix(train_feats, label=train_labels)
dtest = xgb.DMatrix(test_feats, label=test_labels)

if os.path.exists(model_path):
    bst = xgb.Booster(param)
    bst.load_model(model_path)
    print 'load model from '+model_path
else:
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    t1 = time.time()
    print 'training ... ...'
    bst = xgb.train(param, dtrain, num_round, evals=watchlist, early_stopping_rounds=5)
    t2 = time.time()
    print 'training finished in %f seconds' % (t2-t1)
    print 'best iteration is '+str(bst.best_iteration)
    if not os.path.exists('./models/xgbt'):
        os.mkdir('./models/xgbt')
    bst.save_model(model_path)
    print 'model saved to '

pred = bst.predict(dtest)
print 'Average recognition rate is %f' % (sum(pred==test_labels)*1.0/len(pred))
