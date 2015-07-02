#!/usr/bin/env python

import cv2
import sys
import os

COMPCAR_ROOT = '/home/lhy/Documents/Data/CompCars'

assert(len(sys.argv) > 1)

with open(sys.argv[1]) as fd:
    for img_path in fd.readlines():
        img_path = img_path.strip()
        im = cv2.imread(os.path.join(COMPCAR_ROOT, 'image', img_path))
        cv2.imshow("image", im)
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
