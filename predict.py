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
from core.boxParse import BoundingBox
import config.config as cfg
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf


class SSDPredict(object):
    def __init__(self, weight_path):
        self.class_names = cfg.class_names
        self.input_shape = cfg.input_shape
        self.confidence = 0.5

        self.model = SSD300()
        self.model.load_weights(weight_path)
        self.bbox_parse = BoundingBox(cfg.num_classes)

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
        new_image = Image.new('RGB', self.input_shape, (128, 128, 128))
        new_image.paste(image, ((image_w - new_w)//2, (image_h - new_h)//2))

        # 归一化
        image_data = np.array(new_image, dtype=np.float32)
        image_data /= 255.
        image_data = np.clip(image_data, 0.0, 1.0)
        image_data = np.expand_dims(image_data, 0)  # 增加batch的维度

        return image_data

    def ssd_correct_boxes(self, boxes, image_shape):
        # scale = np.min(self.input_shape / image_shape)
        #
        # input_width, input_height = self.input_shape
        # new_width, new_height = image_shape * scale
        #
        # dx = (input_width - new_width) / 2
        # dy = (input_height - new_height) / 2
        #
        # boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
        # boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy

        new_shape = image_shape * np.min(self.input_shape / image_shape)

        offset = (self.input_shape - new_shape) / 2. / self.input_shape
        scale = self.input_shape / new_shape
        top, bottom, left, right = np.expand_dims(boxes[:, 0], axis=-1),\
                                   np.expand_dims(boxes[:, 1], axis=-1),\
                                   np.expand_dims(boxes[:, 2], axis=-1),\
                                   np.expand_dims(boxes[:, 3], axis=-1)

        box_yx = np.concatenate(((top + bottom)/2, (left + right)/2), axis=-1)
        box_hw = np.concatenate((bottom - top, right - left), axis=-1)

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[:, 0:1],
            box_mins[:, 1:2],
            box_maxes[:, 0:1],
            box_maxes[:, 1:2]
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape],axis=-1)

        return boxes

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        resize_image = self.process_image(image)
        pred = self.model(resize_image)

        # 将预测结果进行解码
        result = self.bbox_parse.detection_out(pred, confidence_threshold=self.confidence)

        # 筛选出其中得分高于confidence的框
        top_label = result[:, 0]
        top_conf = result[:, 1]
        top_boxes = result[:, 2:]

        # 因为预测的图片是有灰条，所以需要还原boxes在原图的坐标
        top_boxes = self.ssd_correct_boxes(top_boxes, image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0]

        for i, c in enumerate(top_label):
            c = int(c)
            predicted_class = self.class_names[c - 1]
            score = top_conf[i]

            top, left, bottom, right = top_boxes[i]
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
            print(label)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle([left, top, right, bottom], outline=self.colors[int(c) - 1], width=thickness)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(c) - 1])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


if __name__ == '__main__':
    img_path = "D:/Python_Code/Dataset/VOCdevkit/VOC2012/JPEGImages/2007_000123.jpg"
    # img_path = "street.jpg"

    ssd = SSDPredict('./model/ssd_3.7230.h5')

    if not os.path.exists(img_path):
        print("Error,image path is not exists.")
        exit(-1)

    image = Image.open(img_path)

    image = ssd.detect_image(image)
    image.show()
