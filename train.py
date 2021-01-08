# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/1/4 22:09
# @Software: PyCharm
# @Brief: 训练脚本

import os
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
import config.config as cfg
from nets.ssd import SSD300
from core.loss import MultiboxLoss
from core.boxParse import BoundingBox
from core.anchorGenerate import get_anchors
from core.dataReader import DataReader


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    anchors = get_anchors()
    box_parse = BoundingBox(cfg.num_classes, anchors)
    reader = DataReader(cfg.annotation_path, box_parse, cfg.num_classes, cfg.batch_size, cfg.input_shape)
    train_data = reader.generate('train')
    validation_data = reader.generate('validation')

    train_steps = len(reader.train_lines) // cfg.batch_size
    validation_steps = len(reader.validation_lines) // cfg.batch_size

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(reader.train_lines),
                                                                               len(reader.validation_lines),
                                                                               cfg.batch_size))

    model = SSD300()
    model.compile(optimizer=optimizers.Adam(lr=cfg.learn_rating),
                  loss=MultiboxLoss(cfg.num_classes, neg_pos_ratio=3.0).compute_loss)

    train_by_fit(model, train_data, train_steps, validation_data, validation_steps)


def train_by_fit(model, train_datasets, train_steps, valid_datasets=None, valid_steps=None):
    """
    使用fit方式训练，更规范的添加callbacks参数
    :param model: 训练模型
    :param train_datasets: 训练集数据
    :param valid_datasets: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param valid_steps: 同上
    :return: None
    """
    cbk = [
        callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1),
        callbacks.ModelCheckpoint('./model/ssd_{val_loss:.04f}.h5', save_best_only=True, save_weights_only=True)
    ]

    model.fit(train_datasets,
              steps_per_epoch=train_steps,
              validation_data=valid_datasets,
              validation_steps=valid_steps,
              epochs=cfg.epochs,
              callbacks=cbk)


if __name__ == '__main__':
    main()
