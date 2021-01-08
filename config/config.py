# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/12/26 11:44
# @Software: PyCharm
# @Brief: 配置文件

annotation_path = "./config/train.txt"
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

input_shape = (300, 300)
num_classes = len(class_names) + 1

valid_rate = 0.1
batch_size = 4

epochs = 50
learn_rating = 1e-4

variances = [0.1, 0.1, 0.2, 0.2]
