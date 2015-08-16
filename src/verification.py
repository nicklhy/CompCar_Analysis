import os, sys
import argparse
import time
import cPickle
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import joint_bayes as jb

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CAFFE_ROOT = os.path.join(ROOT_DIR, 'third-parties/caffe')
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

print 'ROOT path is: ', ROOT_DIR

parser = argparse.ArgumentParser(description='verification experiments')
parser.add_argument('--level', dest='level',
                    help='easy, medium or hard',
                    default='easy', type=str)
parser.add_argument('--gpu_id', dest='gpu_id',
                    help='GPU device to use [0]',
                    default=0, type=int)
args = parser.parse_args()

if args.level == 'easy':
    test_file = os.path.join(ROOT_DIR, 'data/train_test_split/verification/verification_pairs_%s.txt' % args.level)
else:
    print 'level can only be easy, medium or hard'
    sys.exit(-1)

gpu_id = 1
model_def = 'models/compcar_model/deploy.prototxt'
model_weights = 'models/compcar_model/bvlc_googlenet_compcar_model_iter_200000.caffemodel'

train_list = map(lambda s: s.strip(), open(os.path.join(ROOT_DIR, 'data/train_test_split/verification/verification_train.txt')).readlines())
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

jb_model_file = os.path.join(ROOT_DIR, 'cache', 'jb_model.pkl')
if os.path.exists(jb_model_file):
    with open(jb_model_file, 'rb') as fd:
        A, G = cPickle.load(fd)
    print 'joint bayes model loaded'
else:
    # extract deep feature and train
    train_feat_file = os.path.join(ROOT_DIR, 'cache', 'verification_train_feat.pkl')
    if not os.path.exists(train_feat_file):
        os.system('cd %s; python src/extract_deep_feature.py --list_file data/train_test_split/verification/verification_train.txt --data_dir data/cropped_image --output cache/verification_train_feat.pkl --model_def models/compcar_model/deploy.prototxt --weights models/compcar_model/bvlc_googlenet_compcar_model_iter_200000.caffemodel --gpu_id 1' % (ROOT_DIR))

    with open(train_feat_file, 'rb') as fd:
        train_list, train_feat = cPickle.load(fd)

    train_label = map(lambda s: int(os.path.dirname(s).split('/')[-2]), train_list)

    print 'Training ...'
    t1 = time.time()
    A, G = jb.train(train_feat, train_label)
    t2 = time.time()
    with open(jb_model_file, 'wb') as fd:
        cPickle.dump([A, G], fd, cPickle.HIGHEST_PROTOCOL)
    print 'Finished in %f seconds' % (t2-t1)


print 'Initializing CNN ...'
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net(model_def, model_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))
net.blobs['data'].reshape(2, 3, 224, 224)

test_list = map(lambda s: s.strip().split(' '), open(test_file).readlines())
N = len(test_list)
preds = np.zeros((N,))
gts = np.array(map(lambda t: int(t[-1]), test_list))
hit_num = 0
threshold = -136
for i, test_data in enumerate(test_list):
    gt = int(test_data[-1])
    t1 = time.time()
    net.blobs['data'].data[0, :, :, :] = transformer.preprocess('data', caffe.io.load_image(os.path.join(ROOT_DIR, 'data', 'cropped_image', test_data[0])))
    net.blobs['data'].data[1, :, :, :] = transformer.preprocess('data', caffe.io.load_image(os.path.join(ROOT_DIR, 'data', 'cropped_image', test_data[1])))
    out = net.forward()
    x1 = net.blobs['pool5/7x7_s1'].data[0].flatten()
    x2 = net.blobs['pool5/7x7_s1'].data[1].flatten()
    t2 = time.time()
    preds[i] = jb.verify(A, G, x1, x2)
    t3 = time.time()
    if (preds[i]>threshold and gt==1) or(preds[i]<threshold and gt==0):
        hit_num = hit_num+1

    if (i+1)%100==0:
        print '***************** image %d *******************' % (i+1)
        print '%d --- Feature extraction in %f s, prediction: %f s\nresult: %f(%d)' % (i+1, t2-t1, t3-t2, preds[i], gt)

print 'pos mean, std, mean, max : %f, %f, %f, %f' % (preds[gts==1].mean(), preds[gts==1].std(), preds[gts==1].min(), preds[gts==1].max())
print 'neg mean, std, mean, max : %f, %f, %f, %f' % (preds[gts==0].mean(), preds[gts==0].std(), preds[gts==0].min(), preds[gts==0].max())
print 'hit rate: %f' % (hit_num*1.0/N)
