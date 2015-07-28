#!/usr/bin/env python

import os
import sys
import scipy.io as sio

if len(sys.argv)>2:
    level = sys.argv[2]
else:
    level = 'make'

DATA_PATH = './data'
make_model_name = sio.loadmat(os.path.join(DATA_PATH, 'misc', 'make_model_name.mat'), squeeze_me = True)
makes = make_model_name['make_names'].tolist()
raw_models = make_model_name['model_names'].tolist()
models = filter(lambda s: isinstance(s, unicode) or isinstance(s, str), raw_models)
rmodel2type = dict()
with open(os.path.join(DATA_PATH, 'misc', 'attributes.txt')) as fd:
    lines = fd.readlines()
    lines = lines[1:]
    for line in lines:
        t = line.strip().split(' ')
        rmodel2type[int(t[0])] = int(t[-1])

if level=='make':
    print 'CLASS_NUM = %d' % len(makes)
elif level=='model':
    print 'CLASS_NUM = %d' % len(models)
elif level=='type':
    print 'CLASS_NUM = %d' % 12
else:
    print 'Wrong level'
    sys.exit(-1)

with open(sys.argv[1]) as lt_fd:
    with open(sys.argv[1].replace('.txt', '_'+level+'.txt'), 'w') as lb_fd:
        for img_path in lt_fd.readlines():
            img_path = img_path.strip()
            make, raw_model, year, img_name = img_path.split('/')
            if level == 'make':
                cls_ind = int(make)-1
            elif level == 'model':
                cls_name = raw_models[int(raw_model)-1]
                cls_ind = models.index(cls_name)
            elif level == 'type':
                cls_ind = rmodel2type[int(raw_model)]-1
                if cls_ind==-1:
                    cls_ind = 12
            lb_fd.write('%s %d\n' % (img_path, cls_ind))
