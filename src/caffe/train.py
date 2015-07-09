#!/usr/bin/env python

import os, sys
import argparse
import time
import numpy as np
# from functools import partial
# import multiprocessing

sys.path.insert(0, 'third-parties/caffe/python')
import caffe
import google.protobuf as pb2


class ModelTrainer:
    """docstring for ModelTrainer"""
    def __init__(self, task, level, solver_prototxt, mean_file = None, pretrained_model=None, gpu_id=0, data_root='./data'):
        if gpu_id >= 0:
            caffe.set_device(args.gpu_id)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.model_dir = os.path.dirname(solver_prototxt)
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            self.solver.net.copy_from(pretrained_model)
        self.solver_param = caffe.proto.caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as fd:
            pb2.text_format.Merge(fd.read(), self.solver_param)
        self.transformer = caffe.io.Transformer({'data': self.solver.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))
        if mean_file:
            self.transformer.set_mean('data', np.load(mean_file))

        self.data_root = data_root

        self.train_batch_size = self.solver.net.blobs['data'].data.shape[0]
        self.test_batch_size = self.solver.test_nets[0].blobs['data'].data.shape[0]

        assert(task in ['all', 'front',
                        'rear', 'side',
                        'front_side', 'rear_side'])
        assert(level in ['make', 'model'])
        if task == 'all':
            self.task_str = ''
        else:
            self.task_str = '_'+task
        self.level_str = '_'+level

        self.train_gt = [x.strip().split(' ') for x in open(os.path.join(data_root, 'train_test_split/classification/train' + self.task_str + self.level_str + '.txt')).readlines()]
        self.test_gt = [x.strip().split(' ') for x in open(os.path.join(data_root, 'train_test_split/classification/test' + self.task_str + self.level_str + '.txt')).readlines()]

    def read_images(self, img_list):
        batch_size, channels, height, width = self.solver.net.blobs['data'].data.shape
        X = np.zeros([len(img_list), channels, height, width], dtype=np.float32)
        for i, img in enumerate(img_list):
            X[i, :, :, :] = self.transformer.preprocess('data', caffe.io.load_image(img))
        return X

    # prepare random or pre-defined batch_size data
    def prepare_batch_data(self, phase='train', idx=None, batch_num=1):
        if phase=='train':
            if idx is not None:
                train_idx = idx
            else:
                train_idx = np.random.permutation(np.arange(len(self.train_gt)))[:self.train_batch_size*batch_num]
            train_list = np.array([os.path.join(self.data_root, 'cropped_image', x[0]) for x in self.train_gt])[train_idx]
            self.train_Y = np.array([int(x[1]) for x in self.train_gt], dtype=np.float32)[train_idx]
            self.train_X = self.read_images(train_list)
            self.solver.net.set_input_arrays(self.train_X, self.train_Y)
        elif phase=='test':
            if idx is not None:
                test_idx = idx
            else:
                test_idx = np.random.permutation(np.arange(len(self.test_gt)))[:self.test_batch_size*batch_num]
            test_list = np.array([os.path.join(self.data_root, 'cropped_image', x[0]) for x in self.test_gt])[test_idx]
            self.test_Y = np.array([int(x[1]) for x in self.train_gt], dtype=np.float32)[test_idx]
            self.test_X = self.read_images(test_list)
            self.solver.test_nets[0].set_input_arrays(self.test_X, self.test_Y)

    def train_model(self):
        t1 = time.time()
        assert(self.solver_param.average_loss>=1)
        while self.solver.iter < self.solver_param.max_iter:
            # t3 = time.time()
            self.prepare_batch_data('train', batch_num = self.solver_param.average_loss)
            # t4 = time.time()
            # print 'read training batch in %f seconds' % (t4-t3)
            self.solver.step(self.solver_param.average_loss)
            if self.solver.iter % self.solver_param.display == 0:
                t2 = time.time()
                print 'speed: {:.3f}s / iter'.format((t2-t1)/self.solver_param.display)
                t1 = t2
            if self.solver.iter % (self.solver_param.test_interval) == 0:
            # if True:
                print '#################### test begin ####################'
                t5 = time.time()
                test_num = len(self.test_gt)
                iter_num = int(np.ceil(test_num*1.0/self.test_batch_size))
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for i in xrange(iter_num):
                    if (i+1)*self.test_batch_size>test_num:
                        ids = np.hstack([np.arange(i*self.test_batch_size, test_num),
                                         np.arange(0, (i+1)*self.test_batch_size%test_num)])
                    else:
                        ids = np.arange(i*self.test_batch_size, (i+1)*self.test_batch_size)
                    self.prepare_batch_data('test', ids)
                    self.solver.test_nets[0].forward()
                    s1 += self.solver.test_nets[0].blobs['loss1/loss1'].data.item()
                    s2 += self.solver.test_nets[0].blobs['loss2/loss1'].data.item()
                    s3 += self.solver.test_nets[0].blobs['loss3/loss3'].data.item()
                s1 /= iter_num
                s2 /= iter_num
                s3 /= iter_num
                print 'loss1/loss1: %f' % s1
                print 'loss2/loss1: %f' % s2
                print 'loss3/loss3: %f' % s3
                t6 = time.time()
                print '#################### test finished in %f seconds ####################' % (t6-t5)
            if self.solver.iter % self.solver_param.snapshot == 0:
                model_path = os.path.join(self.model_dir,
                                          '%d.caffemodel' % self.solver.iter)
                self.solver.net.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a caffe model')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='GPU device to use [0]',
                        default=0, type=int)
    parser.add_argument('--task', dest='task',
                        help='all front rear side front_side rear_side',
                        default='all', type=str)
    parser.add_argument('--level', dest='level',
                        help='make model',
                        default='make', type=str)
    args = parser.parse_args()

    if args.task == 'all':
        task_str = ''
    else:
        task_str = '_'+args.task
    level_str = '_'+args.level
    solver_prototxt = 'models/compcar'+task_str+level_str+'/solver.prototxt'
    pretrained_model = 'models/bvlc_googlenet.caffemodel'

    assert(os.path.exists(solver_prototxt) and
           os.path.exists(pretrained_model))

    trainer = ModelTrainer(args.task,
                           args.level,
                           solver_prototxt,
                           mean_file='data/train_test_split/classification/mean_value.npy',
                           pretrained_model=pretrained_model,
                           gpu_id=args.gpu_id,
                           data_root='data')
    trainer.train_model()
