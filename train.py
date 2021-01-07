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
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    anchors = get_anchors()
    box_parse = BoundingBox(cfg.num_classes, anchors)
    reader = DataReader(cfg.annotation_path, box_parse, cfg.num_classes, cfg.batch_size, cfg.input_shape)
    train_data = reader.generate()
    train_steps = len(reader.train_lines) // cfg.batch_size

    cbk = [
        callbacks.ReduceLROnPlateau(verbose=1),
        callbacks.EarlyStopping(patience=10, verbose=1),
        callbacks.ModelCheckpoint('./model/yolov3_{val_loss:.04f}.h5', save_best_only=True, save_weights_only=True)
    ]

    model = SSD300()
    model.compile(optimizer=optimizers.Adam(lr=cfg.learn_rating),
                  loss=MultiboxLoss(cfg.num_classes, neg_pos_ratio=3.0).compute_loss)

    model.fit(train_data,
              steps_per_epoch=max(1, train_steps),
              epochs=cfg.epochs,
              callbacks=cbk)


if __name__ == '__main__':
    main()