# Some classification experiments evaluated on the newly published CompCar dataset
Created by nicklhy(at gmail dot com)

Dataset reference
@InProceedings{Yang_2015_CVPR,
    author = {Yang, Linjie and Luo, Ping and Change Loy, Chen and Tang, Xiaoou},
    title = {A Large-Scale Car Dataset for Fine-Grained Categorization and Verification},
    journal = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2015}
}

### Introduction

### Requirements
1. [Requirements: software]
2. [Requirements: hardware]

### Requirements: software
1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

    ```make
    # In your Makefile.config, make sure to have this line uncommented
    WITH_PYTHON_LAYER := 1
    ```
You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `python-numpy`, `python-scipy`, `python-matplotlib`, `python-opencv`.
3. For fast-rcnn based classification experiments, [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) is needed.

### Requirements: hardware
1. For training large CNN networks (VGG16, GoogleNet), a good GPU (e.g., Titan, K20, K40, ...) is needed.
2. Other non-deep-learning methods have no specific hardware requirements.
