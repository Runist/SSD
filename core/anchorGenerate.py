# -*- coding: utf-8 -*-
# @File : anchorGenerate.py
# @Author: Runist
# @Time : 2020/12/26 12:52
# @Software: PyCharm
# @Brief: anchor相关


import numpy as np
from nets.ssd_layers import Anchor


class GetAnchor(Anchor):
    def __init__(self, img_size, min_size, max_size, aspect_ratios=None, variances=None,
                 **kwargs):
        """

        :param img_size: 输入到SSD的shape
        :param min_size: 先验框的短边
        :param max_size: 先验框的长边
        :param aspect_ratios: 不同尺度的先验框
        :param variances: 每个框变化尺度
        :param kwargs:
        """
        super().__init__(img_size, min_size, max_size, **kwargs)

        if variances is None:
            variances = [0.1]

        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = []

        if max_size < min_size:
            raise Exception('max_size must be greater than min_size.')

        if aspect_ratios:
            for ar in aspect_ratios:
                self.aspect_ratios.append(ar)
                self.aspect_ratios.append(1 / ar)
        else:
            self.aspect_ratios = [1, 1]

        self.variances = np.array(variances)

        super(Anchor, self).__init__(**kwargs)

    def call(self, input_shape, **kwargs):
        layer_height, layer_width = input_shape
        img_height, img_width = self.img_size

        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            # 前两个缩放系数为1的框，都为正方形
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 缩放系数不为1的anchor，是正交的两个一样的框
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        # 将框偏移至特征层中心
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # linspace(start, stop, num)返回从[start, stop]范围内num个均匀间隔的样本。
        # 这些点就是各个特征层映射到原始图片上的点
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        # 根据x, y序列生成对应矩阵。返回是维度上的坐标，因为特征层只有两维，所以只有x,y的坐标
        centers_x, centers_y = np.meshgrid(linx, liny)

        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        num_priors = len(self.aspect_ratios)
        # 将坐标xy坐标合并
        anchor = np.concatenate((centers_x, centers_y), axis=1)
        # tile(A, reps)沿着reps给定的维度和数量对A进行复制
        anchor = np.tile(anchor, (1, 2 * num_priors))   # 这里是将中心坐标按照先验框的数量进行扩增

        # 获得先验框的左上角和右下角, ::4代表间隔为4，跳着索引
        anchor[:, 0::4] -= box_widths   # x1
        anchor[:, 1::4] -= box_heights  # y1
        anchor[:, 2::4] += box_widths   # x2
        anchor[:, 3::4] += box_heights  # y2

        # 变成小数的形式（现在是在原始图像的尺度，要变成小数，就要除去图像的长宽）
        anchor[:, 0::2] /= img_width    # x坐标
        anchor[:, 1::2] /= img_height   # y坐标
        anchor = anchor.reshape(-1, 4)

        anchor = np.maximum(anchor, 0.0)
        anchor = np.minimum(anchor, 1.0)

        num_boxes = len(anchor)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        # 把variances系数合并到anchor中
        anchor = np.concatenate((anchor, variances), axis=1)

        return anchor


def get_anchors(img_size=(300, 300)):
    if img_size == (512, 512):
        features_map_length = [64, 32, 16, 8, 6, 4]
    elif img_size == (300, 300):
        features_map_length = [38, 19, 10, 5, 3, 1]
    else:
        raise ValueError('Unsupported img_size - `{}`, Use (300,300), (512,512).'.format(img_size))

    net = {}
    anchor = GetAnchor(img_size, min_size=30.0, max_size=60.0, aspect_ratios=[1, 2],
                       variances=[0.1, 0.1, 0.2, 0.2],
                       name='f38_norm_anchor')
    net['f38_norm_anchor'] = anchor([features_map_length[0], features_map_length[0]])

    anchor = GetAnchor(img_size, min_size=60.0, max_size=111.0, aspect_ratios=[1, 2, 3],
                       variances=[0.1, 0.1, 0.2, 0.2],
                       name='f19_anchor')
    net['f19_anchor'] = anchor([features_map_length[1], features_map_length[1]])

    anchor = GetAnchor(img_size, min_size=111.0, max_size=162.0, aspect_ratios=[1, 2, 3],
                       variances=[0.1, 0.1, 0.2, 0.2],
                       name='f10_anchor')
    net['f10_anchor'] = anchor([features_map_length[2], features_map_length[2]])

    anchor = GetAnchor(img_size, min_size=162.0, max_size=213.0, aspect_ratios=[1, 2, 3],
                       variances=[0.1, 0.1, 0.2, 0.2],
                       name='f5_anchor')
    net['f5_anchor'] = anchor([features_map_length[3], features_map_length[3]])

    anchor = GetAnchor(img_size, min_size=213.0, max_size=264.0, aspect_ratios=[1, 2],
                       variances=[0.1, 0.1, 0.2, 0.2],
                       name='f3_anchor')
    net['f3_anchor'] = anchor([features_map_length[4], features_map_length[4]])

    anchor = GetAnchor(img_size, min_size=264.0, max_size=315.0, aspect_ratios=[1, 2],
                       variances=[0.1, 0.1, 0.2, 0.2],
                       name='f1_anchor')

    net['f1_anchor'] = anchor([features_map_length[5], features_map_length[5]])

    net['anchor'] = np.concatenate([net['f38_norm_anchor'],
                                    net['f19_anchor'],
                                    net['f10_anchor'],
                                    net['f5_anchor'],
                                    net['f3_anchor'],
                                    net['f1_anchor']],
                                   axis=0)

    return net['anchor']
