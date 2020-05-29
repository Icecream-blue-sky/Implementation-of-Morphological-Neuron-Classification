import classification as clf
import os
root_path = 'D:\\Study\\The Road to AI\\AIRoad\\neuro_data\\train_dataset'
folder_path = 'D:\\Study\\The Road to AI\\AIRoad\\neuro_data\\train_dataset\\totaldata_measure\\slice'
class_num = 10
clf.preprocess_data2(folder_path,root_path,class_num)