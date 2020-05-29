import classification as clf
root_path = 'D:\\Study\\The Road to AI\\AIRoad\\neuro_data\\train_dataset_test'
class_num = 5
max_level = 24
slice_max_depth = 10
slice_max_size = 20
clf.decompose(root_path,class_num,max_level,slice_max_depth,slice_max_size)