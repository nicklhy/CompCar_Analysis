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
Caffe build with mkl, cudnn is strongly recommended.
2. For fast-rcnn based classification experiments, [fast-rcnn](https://github.com/rbgirshick/fast-rcnn) is needed.
3. For xgboost based experiments, [xgboost](https://github.com/dmlc/xgboost) is needed.
4. Python packages you might not have: `python-numpy`, `python-scipy`, `python-matplotlib`, `python-opencv`, `python-scikit-learn`.

### Requirements: hardware
1. For training large CNN networks (VGG16, GoogleNet), a good GPU (e.g., Titan, K20, K40, ...) is needed.
2. Other non-deep-learning methods have no specific hardware requirements.

### Instructions
1. Prepare the dataset
Download the CompCar dataset at any place in you hard disk and build a soft link to our repo’s root directory as the name `data`:

    ```
    ln -s /path/to/CompCar /path/to/CompCar_Analysis/data
    ```

2. Split and transform the original dataset into some specific forms
To split the vehicle images for training and testing into different angles, use tools/split_viewpoints.py

    ```
    ./tools/split_viewpoints.py
    ```

To crop all vehicles from the original images

    ```
    ./tools/generate_cropped_image.py 4
    # the argment `4` is the process num we use to accelerate the program, default is 2(multi-thread is useless in Python, thus, we choose multi-process).
    ```

To generate lmdb data for `caffe`
    * ./tools/generate_label_list.py will generate the label included list files for caffe, but it will be called in ./tools/generate_lmdb.sh which means you do not need to run it yourself.

    ```
    ./tools/generate_label_list.py data/train_test_split/classification/train.txt make
    # You can substitute `train.txt` to `${phase}${_viewpoint}.txt`({phase: [train, test], _viewpoints: [``, `_front`, `_rear`, `_side`, `_front_side`, `_rear_side`]}) and subsitute `make` to `${level}`({level: [make, model]}), which will generate a new `phase_viewpoint_level.txt` list file with labels after the image name.
    ```

    * ./tools/generate_lmdb.sh will generate the lmdb data of different phases, viewpoints and level. Just run it in the root directory of repo `CompCar_Analysis`.

    ```
    ./tools/generate_lmdb.sh
    ```

