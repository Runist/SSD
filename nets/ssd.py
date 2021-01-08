# -*- coding: utf-8 -*-
# @File : ssd.py
# @Author: Runist
# @Time : 2020/12/15 19:55
# @Software: PyCharm
# @Brief:
import config.config as cfg
from tensorflow.keras import layers, models
from nets.vgg import VGG16
from nets.ssd_layers import Normalize, Anchor


def SSD300():
    # 300,300,3
    height, width = cfg.input_shape
    input_tensor = layers.Input(shape=(height, width, 3))

    # SSD结构,net字典
    net = VGG16(input_tensor)

    # -----------------------将提取到的主干特征进行处理---------------------------#
    # 对38x38大小的特征层进行处理 38,38,512
    # 因为f38卷积层数较少，特征不够明显，所以就进行归一化操作，使得预测效果更好
    net['f38_norm'] = Normalize(20, name='f38_norm')(net['f38'])
    num_anchors = 4
    # 预测框的处理
    # num_anchors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['f38_loc'] = layers.Conv2D(num_anchors * 4,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   name='f38_loc')(net['f38_norm'])
    net['f38_loc_flat'] = layers.Flatten(name='f38_loc_flat')(net['f38_loc'])
    # num_anchors表示每个网格点先验框的数量，num_classes是所分的类
    net['f38_norm_conf'] = layers.Conv2D(num_anchors * cfg.num_classes,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         name='f38_norm_conf')(net['f38_norm'])
    net['f38_norm_conf_flat'] = layers.Flatten(name='f38_norm_conf_flat')(net['f38_norm_conf'])
    anchor = Anchor(cfg.input_shape, min_size=30.0, max_size=60.0, aspect_ratios=[1, 2],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='f38_norm_anchor')

    net['f38_norm_anchor'] = anchor(net['f38_norm'])

    # 对f19层进行处理
    num_anchors = 6
    net['f19_loc'] = layers.Conv2D(num_anchors * 4,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   name='f19_loc')(net['f19'])
    net['f19_loc_flat'] = layers.Flatten(name='f19_loc_flat')(net['f19_loc'])
    net['f19_conf'] = layers.Conv2D(num_anchors * cfg.num_classes,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name='f19_conf')(net['f19'])
    net['f19_conf_flat'] = layers.Flatten(name='f19_conf_flat')(net['f19_conf'])

    anchor = Anchor(cfg.input_shape, min_size=60.0, max_size=111.0, aspect_ratios=[1, 2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='f19_anchor')
    net['f19_anchor'] = anchor(net['f19'])

    # 对f10进行处理
    num_anchors = 6
    net['f10_loc'] = layers.Conv2D(num_anchors * 4,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   name='f10_loc')(net['f10'])
    net['f10_loc_flat'] = layers.Flatten(name='f10_loc_flat')(net['f10_loc'])
    net['f10_conf'] = layers.Conv2D(num_anchors * cfg.num_classes,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name='f10_conf')(net['f10'])
    net['f10_conf_flat'] = layers.Flatten(name='f10_conf_flat')(net['f10_conf'])

    anchor = Anchor(cfg.input_shape, min_size=111.0, max_size=162.0, aspect_ratios=[1, 2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='f10_anchor')
    net['f10_anchor'] = anchor(net['f10'])

    # 对f5进行处理
    num_anchors = 6
    net['f5_loc'] = layers.Conv2D(num_anchors * 4,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  name='f5_loc')(net['f5'])
    net['f5_loc_flat'] = layers.Flatten(name='f5_loc_flat')(net['f5_loc'])
    net['f5_conf'] = layers.Conv2D(num_anchors * cfg.num_classes,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   name='f5_conf')(net['f5'])
    net['f5_conf_flat'] = layers.Flatten(name='f5_conf_flat')(net['f5_conf'])

    anchor = Anchor(cfg.input_shape, min_size=162.0, max_size=213.0, aspect_ratios=[1, 2, 3],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='f5_anchor')
    net['f5_anchor'] = anchor(net['f5'])

    # 对f3进行处理
    num_anchors = 4
    net['f3_loc'] = layers.Conv2D(num_anchors * 4,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  name='f3_loc')(net['f3'])
    net['f3_loc_flat'] = layers.Flatten(name='f3_loc_flat')(net['f3_loc'])
    net['f3_conf'] = layers.Conv2D(num_anchors * cfg.num_classes,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   name='f3_conf')(net['f3'])
    net['f3_conf_flat'] = layers.Flatten(name='f3_conf_flat')(net['f3_conf'])

    anchor = Anchor(cfg.input_shape, min_size=213.0, max_size=264.0, aspect_ratios=[1, 2],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='f3_anchor')
    net['f3_anchor'] = anchor(net['f3'])

    # 对f1进行处理
    num_anchors = 4
    net['f1_loc'] = layers.Conv2D(num_anchors * 4,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  name='f1_loc')(net['f1'])
    net['f1_loc_flat'] = layers.Flatten(name='f1_loc_flat')(net['f1_loc'])
    net['f1_conf'] = layers.Conv2D(num_anchors * cfg.num_classes,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   name='f1_conf')(net['f1'])
    net['f1_conf_flat'] = layers.Flatten(name='f1_conf_flat')(net['f1_conf'])

    anchor = Anchor(cfg.input_shape, min_size=264.0, max_size=315.0, aspect_ratios=[1, 2],
                    variances=[0.1, 0.1, 0.2, 0.2],
                    name='f1_anchor')

    net['f1_anchor'] = anchor(net['f1'])

    # 将所有结果进行堆叠
    # 坐标变换参数
    net['loc'] = layers.concatenate([net['f38_loc_flat'],
                                     net['f19_loc_flat'],
                                     net['f10_loc_flat'],
                                     net['f5_loc_flat'],
                                     net['f3_loc_flat'],
                                     net['f1_loc_flat']],
                                    axis=1, name='loc')
    # 置信度
    net['conf'] = layers.concatenate([net['f38_norm_conf_flat'],
                                      net['f19_conf_flat'],
                                      net['f10_conf_flat'],
                                      net['f5_conf_flat'],
                                      net['f3_conf_flat'],
                                      net['f1_conf_flat']],
                                     axis=1, name='conf')
    # 先验框
    net['anchor'] = layers.concatenate([net['f38_norm_anchor'],
                                        net['f19_anchor'],
                                        net['f10_anchor'],
                                        net['f5_anchor'],
                                        net['f3_anchor'],
                                        net['f1_anchor']],
                                       axis=1, name='anchor')

    num_boxes = net['loc'].shape[-1] // 4

    # 8732,4
    net['loc'] = layers.Reshape((num_boxes, 4), name='loc_final')(net['loc'])
    # 8732,21
    net['conf'] = layers.Reshape((num_boxes, cfg.num_classes), name='conf_logits')(net['conf'])
    net['conf'] = layers.Activation('softmax', name='conf_final')(net['conf'])

    net['predictions'] = layers.concatenate([net['loc'],
                                             net['conf'],
                                             net['anchor']],
                                            axis=2, name='predictions')

    model = models.Model(net['input'], net['predictions'])
    # model.summary()

    return model


