# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2021/1/6 20:41
# @Software: PyCharm
# @Brief: 预测脚本
import os
import numpy as np
import colorsys
from nets.ssd import SSD300
import config.config as cfg
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf


class SSD(object):
    def __init__(self, weight_path, ):
        self.class_names = cfg.label
        self.input_shape = cfg.input_shape
        self.confidence = 0.5

        model = SSD300()
        self.model = model.load_weights(weight_path)

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def process_image(self, image):
        """
        读取图片，填充图片后归一化
        :param image: 图片路径
        :return: 图片的np数据、宽、高
        """
        # 获取原图尺寸 和 网络输入尺寸
        image_w, image_h = image.size
        input_w, input_h = self.input_shape

        scale = min(input_w / image_w, input_h / image_h)
        new_w = int(image_w * scale)
        new_h = int(image_h * scale)

        # 插值变换、填充图片
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # 归一化
        image_data = np.array(image, dtype=np.float32)
        image_data /= 255.
        image_data = np.clip(image_data, 0.0, 1.0)
        image_data = np.expand_dims(image_data, 0)  # 增加batch的维度

        return image_data

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        resize_image = self.process_image(image)

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[c - 1]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle([left, top, right, bottom], outline=self.colors[int(c) - 1], width=thickness)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c) - 1])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
