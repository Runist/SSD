# -*- coding: utf-8 -*-
# @File : vgg.py
# @Author: Runist
# @Time : 2020/12/14 22:20
# @Software: PyCharm
# @Brief:

from tensorflow.keras import layers


def VGG16(input_tensor):
    net = {'input': input_tensor}
    # Block 1
    # 300,300,3 -> 150,150,64
    net['conv1_1'] = layers.Conv2D(64,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = layers.Conv2D(64,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = layers.MaxPooling2D((2, 2),
                                       strides=(2, 2),
                                       padding='same',
                                       name='pool1')(net['conv1_2'])

    # Block 2
    # 150,150,64 -> 75,75,128
    net['conv2_1'] = layers.Conv2D(128,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = layers.Conv2D(128,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = layers.MaxPooling2D((2, 2),
                                       strides=(2, 2),
                                       padding='same',
                                       name='pool2')(net['conv2_2'])
    # Block 3
    # 75,75,128 -> 38,38,256
    net['conv3_1'] = layers.Conv2D(256,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = layers.Conv2D(256,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = layers.Conv2D(256,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = layers.MaxPooling2D((2, 2),
                                       strides=(2, 2),
                                       padding='same',
                                       name='pool3')(net['conv3_3'])
    # Block 4
    # 38,38,256 -> 19,19,512
    net['conv4_1'] = layers.Conv2D(512,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = layers.Conv2D(512, 
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['f38'] = layers.Conv2D(512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='f38')(net['conv4_2'])
    net['pool4'] = layers.MaxPooling2D((2, 2),
                                       strides=(2, 2),
                                       padding='same',
                                       name='pool4')(net['f38'])
    # Block 5
    # 19,19,512 -> 19,19,512
    net['conv5_1'] = layers.Conv2D(512,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = layers.Conv2D(512,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = layers.Conv2D(512,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = layers.MaxPooling2D((3, 3), 
                                       strides=(1, 1),
                                       padding='same',
                                       name='pool5')(net['conv5_3'])
    # conv6
    # 19,19,512 -> 19,19,1024
    net['conv6'] = layers.Conv2D(1024,
                                 kernel_size=(3, 3),
                                 dilation_rate=(6, 6),
                                 activation='relu',
                                 padding='same',
                                 name='conv6')(net['pool5'])

    # x = Dropout(0.5, name='drop6')(x)
    # f19
    # 19,19,1024 -> 19,19,1024
    net['f19'] = layers.Conv2D(1024,
                               kernel_size=(1, 1),
                               activation='relu',
                               padding='same',
                               name='f19')(net['conv6'])

    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    # 19,19,512 -> 10,10,512
    net['conv7'] = layers.Conv2D(256,
                                 kernel_size=(1, 1),
                                 activation='relu',
                                 padding='same',
                                 name='conv7')(net['f19'])
    net['f10'] = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(net['conv7'])
    net['f10'] = layers.Conv2D(512,
                               kernel_size=(3, 3),
                               strides=(2, 2),
                               activation='relu',
                               name='f10')(net['f10'])

    # Block 7
    # 10,10,512 -> 5,5,256
    net['conv8'] = layers.Conv2D(128,
                                 kernel_size=(1, 1),
                                 activation='relu',
                                 padding='same',
                                 name='conv8')(net['f10'])
    net['f5'] = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(net['conv8'])
    net['f5'] = layers.Conv2D(256,
                              kernel_size=(3, 3),
                              strides=(2, 2),
                              activation='relu',
                              padding='valid',
                              name='f5')(net['f5'])
    # Block 8
    # 5,5,256 -> 3,3,256
    net['conv9'] = layers.Conv2D(128,
                                 kernel_size=(1, 1),
                                 activation='relu',
                                 padding='same',
                                 name='conv9')(net['f5'])
    net['f3'] = layers.Conv2D(256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation='relu',
                              padding='valid',
                              name='f3')(net['conv9'])

    # Block 9
    # 3,3,256 -> 1,1,256
    net['conv10'] = layers.Conv2D(128,
                                  kernel_size=(1, 1),
                                  activation='relu',
                                  padding='same',
                                  name='conv10')(net['f3'])
    net['f1'] = layers.Conv2D(256,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation='relu',
                              padding='valid',
                              name='f1')(net['conv10'])

    return net
