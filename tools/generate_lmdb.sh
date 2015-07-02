############################### make train ###############################
# ./generate_label_list.py ../train_test_split/classification/train.txt make
# ./generate_label_list.py ../train_test_split/classification/train_front.txt make
# ./generate_label_list.py ../train_test_split/classification/train_rear.txt make
# ./generate_label_list.py ../train_test_split/classification/train_side.txt make
# ./generate_label_list.py ../train_test_split/classification/train_front_side.txt make
# ./generate_label_list.py ../train_test_split/classification/train_rear_side.txt make


# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_make.txt train_make_lmdb -resize_width=224 -resize_height=224 -check_size
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_front_make.txt train_front_make_lmdb -resize_width=224 -resize_height=224 -check_size
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_rear_make.txt train_rear_make_lmdb -resize_width=224 -resize_height=224 -check_size
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_side_make.txt train_side_make_lmdb -resize_width=224 -resize_height=224 -check_size
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_front_side_make.txt train_front_side_make_lmdb -resize_width=224 -resize_height=224 -check_size
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_rear_side_make.txt train_rear_side_make_lmdb -resize_width=224 -resize_height=224 -check_size

############################### make train ###############################
# ./generate_label_list.py ../train_test_split/classification/test.txt make
# ./generate_label_list.py ../train_test_split/classification/test_front.txt make
# ./generate_label_list.py ../train_test_split/classification/test_rear.txt make
# ./generate_label_list.py ../train_test_split/classification/test_side.txt make
# ./generate_label_list.py ../train_test_split/classification/test_front_side.txt make
# ./generate_label_list.py ../train_test_split/classification/test_rear_side.txt make


# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_make.txt test_make_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_front_make.txt test_front_make_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_rear_make.txt test_rear_make_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_side_make.txt test_side_make_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_front_side_make.txt test_front_side_make_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_rear_side_make.txt test_rear_side_make_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true



############################### model train ###############################
./generate_label_list.py ../train_test_split/classification/train.txt model
./generate_label_list.py ../train_test_split/classification/train_front.txt model
./generate_label_list.py ../train_test_split/classification/train_rear.txt model
./generate_label_list.py ../train_test_split/classification/train_side.txt model
./generate_label_list.py ../train_test_split/classification/train_front_side.txt model
./generate_label_list.py ../train_test_split/classification/train_rear_side.txt model


/home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_model.txt train_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
/home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_front_model.txt train_front_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
/home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_rear_model.txt train_rear_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
/home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_side_model.txt train_side_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
/home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_front_side_model.txt train_front_side_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
/home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/train_rear_side_model.txt train_rear_side_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true

############################### model test ###############################
# ./generate_label_list.py ../train_test_split/classification/test.txt model
# ./generate_label_list.py ../train_test_split/classification/test_front.txt model
# ./generate_label_list.py ../train_test_split/classification/test_rear.txt model
# ./generate_label_list.py ../train_test_split/classification/test_side.txt model
# ./generate_label_list.py ../train_test_split/classification/test_front_side.txt model
# ./generate_label_list.py ../train_test_split/classification/test_rear_side.txt model


# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_model.txt test_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_front_model.txt test_front_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_rear_model.txt test_rear_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_side_model.txt test_side_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_front_side_model.txt test_front_side_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
# /home/lhy/Documents/Codes/Libs/caffe/build/tools/convert_imageset /home/lhy/Documents/Data/CompCars/cropped_image/ ../train_test_split/classification/test_rear_side_model.txt test_rear_side_model_lmdb -resize_width=224 -resize_height=224 -check_size -shuffle true
