# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/12/26 11:28
# @Software: PyCharm
# @Brief:


import tensorflow as tf
import numpy as np
import config.config as cfg
from PIL import Image
import cv2 as cv


class DataReader(object):
    def __init__(self, data_path, box_parse, num_classes, batch_size, input_shape):
        """
        :param data_path: 图片-标签 对应关系的txt文本路径
        :param box_parse: box解析类对象
        :param batch_size: 批处理数量
        :param input_shape: 输入层的宽高信息
        """
        self.data_path = data_path
        self.box_parse = box_parse
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes - 1
        self.train_lines, self.validation_lines = self.read_data_and_split_data()

    def read_data_and_split_data(self):
        """
        读取图片的路径信息，并按照比例分为训练集和测试集
        :return:
        """
        with open(self.data_path, "r", encoding='utf-8') as f:
            files = f.readlines()

        split = int(cfg.valid_rate * len(files))

        train = files[split:]
        validation = files[:split]

        return train, validation

    def get_data(self, annotation_line):
        """
        获取数据（不增强）
        :param annotation_line: 一行数据（图片路径 + 坐标）
        :return: image，box_data
        """
        line = annotation_line.split()
        image = Image.open(line[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image_width, image_height = image.size
        input_width, input_height = self.input_shape
        scale = min(input_width / image_width, input_height / image_height)

        new_width = int(image_width * scale)
        new_height = int(image_height * scale)

        image = image.resize((new_width, new_height), Image.BICUBIC)
        new_image = Image.new('RGB', self.input_shape, (128, 128, 128))
        new_image.paste(image, ((input_width - new_width)//2, (input_height - new_height)//2))

        image = np.asarray(new_image) / 255

        dx = (input_width - new_width) / 2
        dy = (input_height - new_height) / 2

        # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
        if len(box) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy

        # 去除无效数据
        box_data = np.array(box, dtype='float32')

        # 将bbox的坐标变0-1
        box_data[:, 0] = box_data[:, 0] / input_width
        box_data[:, 1] = box_data[:, 1] / input_height
        box_data[:, 2] = box_data[:, 2] / input_width
        box_data[:, 3] = box_data[:, 3] / input_height

        return image, box_data

    def get_random_data(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        数据增强（改变长宽比例、大小、亮度、对比度、颜色饱和度）
        :param annotation_line: 一行数据
        :param hue: 色调抖动
        :param sat: 饱和度抖动
        :param val: 明度抖动
        :return: image, box_data
        """
        line = annotation_line.split()
        image = Image.open(line[0])

        image_width, image_height = image.size
        input_width, input_height = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 随机生成缩放比例，缩小或者放大
        scale = rand(0.5, 1.5)
        # 随机变换长宽比例
        new_ar = input_width / input_height * rand(0.7, 1.3)

        if new_ar < 1:
            new_height = int(scale * input_height)
            new_width = int(new_height * new_ar)
        else:
            new_width = int(scale * input_width)
            new_height = int(new_width / new_ar)

        image = image.resize((new_width, new_height), Image.BICUBIC)

        dx = rand(0, (input_width - new_width))
        dy = rand(0, (input_height - new_height))
        new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
        new_image.paste(image, (int(dx), int(dy)))
        image = new_image

        # 翻转图片
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 图像增强
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv.cvtColor(np.array(image, np.float32)/255, cv.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = cv.cvtColor(x, cv.COLOR_HSV2RGB)

        # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
        if len(box) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        box[:, [0, 2]] = box[:, [0, 2]] * new_width / image_width + dx
        box[:, [1, 3]] = box[:, [1, 3]] * new_height / image_height + dy
        # 若翻转了图像，框也需要翻转
        if flip:
            box[:, [0, 2]] = input_width - box[:, [2, 0]]

        # 定义边界
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > input_width] = input_width
        box[:, 3][box[:, 3] > input_height] = input_height

        # 计算新的长宽
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        # 去除无效数据
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        box_data = np.array(box, dtype='float32')

        # 将bbox的坐标变0-1
        box_data[:, 0] = box_data[:, 0] / input_width
        box_data[:, 1] = box_data[:, 1] / input_height
        box_data[:, 2] = box_data[:, 2] / input_width
        box_data[:, 3] = box_data[:, 3] / input_height

        return image, box_data

    def generate(self, mode):
        """
        数据生成器
        :return: image, rpn训练标签， 真实框数据
        """

        i = 0
        if mode == 'train':
            n = len(self.train_lines)
        else:
            n = len(self.validation_lines)

        while True:
            image_data = []
            boxes_and_label = []

            if i == 0:
                if mode == 'train':
                    np.random.shuffle(self.train_lines)
                else:
                    np.random.shuffle(self.validation_lines)

            j = 0
            while j < self.batch_size:
                if mode == 'train':
                    image, bbox = self.get_random_data(self.train_lines[i])
                else:
                    image, bbox = self.get_data(self.validation_lines[i])
                i = (i + 1) % n

                if len(bbox) == 0:
                    continue

                j += 1

                one_hot_label = np.eye(self.num_classes)[np.array(bbox[:, 4], np.int32)]
                boxes = np.concatenate([bbox[:, :4], one_hot_label], axis=-1)

                boxes = self.box_parse.assign_boxes(boxes)

                image_data.append(image)
                boxes_and_label.append(boxes)

            image_data = np.array(image_data)
            boxes_and_label = np.array(boxes_and_label, dtype=np.float32)

            yield image_data, boxes_and_label


def rand(small=0., big=1.):
    return np.random.rand() * (big - small) + small


def iou(box_a, box_b):
    """
    根据输入的两个框的坐标，计算iou，
    :param box_a: 第一个框的坐标
    :param box_b: 第二个框的坐标
    :return: iou
    """
    # 如果出现左上角的坐标大过右下角的坐标，则返回iou为0
    if box_a[0] >= box_a[2] or box_a[1] >= box_a[3] or box_b[0] >= box_b[2] or box_b[1] >= box_b[3]:
        return 0.0

    x = max(box_a[0], box_b[0])
    y = max(box_a[1], box_b[1])
    w = min(box_a[2], box_b[2]) - x
    h = min(box_a[3], box_b[3]) - y

    if w < 0 or h < 0:
        # 2个框不相交，分子为0，iou = 0
        return 0.0

    # 计算相交面积
    intersect_area = w * h
    # 计算并集面积
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersect_area

    return intersect_area / union_area

