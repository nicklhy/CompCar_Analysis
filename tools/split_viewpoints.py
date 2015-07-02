#!/usr/bin/env python

import os


def split_viewpoints(compcar_root, list_file):
    splitted_img_list = dict()
    with open(list_file, 'r') as fd:
        img_list = map(lambda s: s.strip(), fd.readlines())

    for img in img_list:
        label_file = img.replace('jpg', 'txt')
        with open(os.path.join(COMPCAR_ROOT, 'label', label_file)) as lb_fd:
            lines = lb_fd.readlines()
            assert(len(lines) == 3)
            vp = int(lines[0].strip())
            if vp not in splitted_img_list:
                splitted_img_list[vp] = []
            splitted_img_list[vp].append(img)

    return splitted_img_list


if __name__ == '__main__':
    COMPCAR_ROOT = './data'
    for f_name in ['train.txt', 'test.txt']:
        list_file = os.path.join(COMPCAR_ROOT,
                                 'train_test_split',
                                 'classification',
                                 f_name)
        splitted_imgs = split_viewpoints(COMPCAR_ROOT, list_file)
        vp_str = {-1: 'unknown',
                  1: 'front',
                  2: 'rear',
                  3: 'side',
                  4: 'front_side',
                  5: 'rear_side'}
        for vp in splitted_imgs:
            with open(list_file.replace('.txt', '_'+vp_str[vp]+'.txt'), 'w') as lt_fd:
                for img in splitted_imgs[vp]:
                    lt_fd.write('%s\n' % img)
