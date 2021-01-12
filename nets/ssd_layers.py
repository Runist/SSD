# -*- coding: utf-8 -*-
# @File : ssd_layers.py
# @Author: Runist
# @Time : 2020/12/16 20:43
# @Software: PyCharm
# @Brief:

from tensorflow.keras import backend, layers
import numpy as np


class Normalize(layers.Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale

        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [layers.InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = backend.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.train_weights = [self.gamma]

    def call(self, x, mask=None):
        output = backend.l2_normalize(x, self.axis)
        output *= self.gamma

        return output


class Anchor(layers.Layer):
    def __init__(self, img_size, min_size, max_size, aspect_ratios=None, **kwargs):
        """

        :param img_size: 输入到SSD的shape
        :param min_size: 先验框的短边
        :param max_size: 先验框的长边
        :param aspect_ratios: 不同尺度的先验框
        :param kwargs:
        """

        super().__init__(**kwargs)

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

        super(Anchor, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """
        计算输出shape
        :param input_shape:
        :return:
        """
        num_priors = len(self.aspect_ratios)
        layer_height, layer_width = input_shape[1: 3]
        num_boxes = num_priors * layer_width * layer_height

        return None, num_boxes, 8

    def call(self, x, **kwargs):
        layer_height, layer_width = x.shape[1:3]
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
        anchor = np.expand_dims(anchor, 0)

        # 从(1, num_anchors, 4)变成(None, num_anchors, 4)
        pattern = [backend.shape(x)[0], 1, 1]
        anchor = backend.tile(anchor, pattern)
        anchor = backend.cast_to_floatx(anchor)

        return anchor
