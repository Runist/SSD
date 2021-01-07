# -*- coding: utf-8 -*-
# @File : loss.py
# @Author: Runist
# @Time : 2021/1/4 22:33
# @Software: PyCharm
# @Brief: loss相关定义
import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def l1_smooth_loss(self, y_true, y_pred):
        """
        计算L1_smooth_loss
        :param y_true: (batch_size, 8732, 4)
        :param y_pred: (batch_size, 8732, 4)
        :return:
        """
        x = y_true - y_pred
        abs_x = tf.abs(x)

        l1_loss = tf.where(tf.less(abs_x, 1.0),
                           0.5 * x ** 2,
                           abs_x - 0.5)

        # 对最后一维度求平均
        return tf.reduce_sum(l1_loss, axis=-1)

    def softmax_loss(self, y_true, y_pred):
        # 因为网络输出带有softmax，所以不能用tf的API直接计算
        y_pred = tf.maximum(y_pred, 1e-7)
        softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        # 总boxes的数量8732
        num_boxes = tf.cast(tf.shape(y_true)[1], tf.float32)

        # 计算所有的loss
        # batch_size,8732,21 -> batch_size,8732
        conf_loss = self.softmax_loss(y_true[:, :, 4:-8], y_pred[:, :, 4:-8])
        # batch_size,8732,4 -> batch_size,8732
        loc_loss = self.l1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])

        # 获取所有的正标签的loss
        # 获取y_true正例个数
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        # 每一张图的正例loc_loss, shape: (batch_size, 8732)
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8], axis=-1)
        # 每一张图的正例conf_loss
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8], axis=-1)

        # 获取一定的负样本
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)

        # 获取大于0的索引
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # 获得一个1.0
        has_min = tf.cast(tf.reduce_any(pos_num_neg_mask), tf.float32)
        num_neg = tf.concat(axis=0, values=[num_neg,
                                            [(1 - has_min) * self.negatives_for_hard]])

        # 注意，由于正样本的数量是在一个batch中获取的，所以负样本的数量也需要在batch中被平均
        # 求平均每个图片要取多少个负样本
        num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.cast(num_neg_batch, tf.int32)

        # 找到实际上在该位置不应该有预测结果的框，求他们最大的置信度。
        max_confs = tf.reduce_max(y_pred[:, :, 5:-8], axis=-1)

        # 取top_k个置信度，将y_true的置信度翻转，只要背景的置信度，从分数高到低依次向下取，作为负样本
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=num_neg_batch)

        # 找到其在1维上的索引
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        # 得到其在同一batch下不同image下的index，shape: (batch_size * num_neg_batch, )
        batch_idx = (tf.reshape(batch_idx, [-1]) * tf.cast(num_boxes, tf.int32) +
                     tf.reshape(indices, [-1]))

        # conf_loss reshape成一维的向量后，方便索引，取出负例的置信度损失
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), batch_idx)
        neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=-1)

        # loss is sum of positives and negatives

        # num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        num_pos = tf.maximum(num_pos, tf.ones_like(num_pos))
        total_conf_loss = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss)
        total_conf_loss /= tf.reduce_sum(num_pos)
        total_loc_loss = tf.reduce_sum(self.alpha * pos_loc_loss) / tf.reduce_sum(num_pos)

        total_loss = total_conf_loss + total_loc_loss

        return total_loss


