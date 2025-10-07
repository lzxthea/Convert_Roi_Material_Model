# encoding:utf-8

import tensorflow as tf
import math


def get_vars_in_scope(name_scope="model_bias"):
    """
    返回bias参数
    """
    # print([v.name for v in tf.compat.v1.trainable_variables(
    #     scope=name_scope
    # )])
    return tf.compat.v1.trainable_variables(
        scope=name_scope
    )


def get_vars_not_in_scope(name_scope="model_bias"):
    # print([v.name for v in tf.compat.v1.trainable_variables() if not v.name.startswith(name_scope)])
    return [v for v in tf.compat.v1.trainable_variables() if not v.name.startswith(name_scope)]


def huber_loss(labels, pred, logging_hook_di, delta=1.0):
    """
    huber loss调整
    Args:
        labels:
        pred:
        delta:

    Returns:

    """
    logging_hook_di['huber_labels'] = labels
    logging_hook_di['huber_pred'] = pred
    residual = tf.abs(pred - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    logging_hook_di['huber_loss'] = large_res
    return tf.where(condition, small_res, large_res)


def weighted_cross_entropy_with_logits(labels, pred):
    """

    Returns:

    """
    binary_label = tf.where(
        labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    )
    wce_loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=tf.reshape(binary_label, [-1]),
        logits=pred,
        pos_weight=tf.reshape(1 + labels, [-1])
    )
    return wce_loss

def weighted_cross_entropy(labels, pred, weight, logging_hook_di):
    """

    Returns:

    """
    logging_hook_di['task_label'] = labels
    logging_hook_di['task_score'] = pred

    binary_label = tf.where(
        labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    )

    # logging_hook_di['binary_label'] = binary_label
    
    wce_loss = -binary_label * tf.log(pred) * weight - (1 - binary_label) * tf.log(1 - pred)

    logging_hook_di['wce_loss'] = wce_loss
    
    return wce_loss

def focal_loss(labels, pred):
    # Focal Loss公式（alpha为正负样本权重，gamma为难度系数，gamma>0）
    alpha = 0.25
    gamma = 2.0
    focal_loss = -alpha * labels * tf.log(pred) * (1 - pred)**gamma \
             - (1 - alpha) * (1 - labels) * tf.log(1 - pred) * pred**gamma
    return focal_loss

def cross_entropy(labels, pred, logging_hook_di):
    """

    Returns:

    """
    # logging_hook_di['task_label'] = labels
    
    # binary_label = tf.where(
    #     labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    # )

    # logging_hook_di['binary_label'] = labels
    # logging_hook_di['binary_pred'] = pred
    ce_loss = -labels * tf.math.log(pred) - (1 - labels) * tf.math.log(1 - pred)

    # logging_hook_di['ce_loss'] = ce_loss

    return ce_loss

def zlin(labels, pred, logging_hook_di):
    logging_hook_di['raw_label'] = labels
    
    binary_label = tf.where(
        labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels)
    )

    return zlin_loss

def binarized_reward(reward, cost_level_mask, key_to_mask, threshold=None):
    """
    对reward的结果进行二值化
    Args:
        reward:
        cost_level_mask:
        threshold:

    Returns:

    """
    high_mask = key_to_mask['high']
    medium_mask = key_to_mask['medium']
    if threshold is None:
        threshold = [11.2, 16.5]
    assert len(threshold) == 2, "threshold: {} len != 2".format(threshold)
    assert threshold[0] <= threshold[1], "threshold: {} must le threshold: {} ".format(
        threshold[0], threshold[1]
    )
    medium_reward = tf.cast(tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask), tf.greater_equal(reward, threshold[0])
    ), tf.float32)
    high_reward = tf.cast(tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask), tf.greater_equal(reward, threshold[1])
    ), tf.float32)
    merge_reward = high_reward + medium_reward
    tf.summary.histogram(
        "medium_reward", tf.boolean_mask(
            merge_reward,
            tf.equal(cost_level_mask, medium_mask)
        )
    )
    tf.summary.histogram(
        "high_reward", tf.boolean_mask(
            merge_reward,
            tf.equal(cost_level_mask, high_mask)
        )
    )
    return merge_reward


def reward_shaping(reward, cost_level_mask, bill_ratio, key_to_mask):
    """
    预测得阈值，价值扶持系数，不同cost_level
    Args:
        reward:
        cost_level_mask: 每个样本映射到不同的level:
            high, medium映射到对应 cost_level_info
        bill_ratio: 计费比:
            (0.0, 0.8]: 低估
            0.8-1.2: 正常
            >1.2 高估
            0.0: 空跑
        key_to_mask:  dict
            high: xx
            medium: xx
    Returns:
    """
    high_mask = key_to_mask['high']
    medium_mask = key_to_mask['medium']
    is_medium_cost_level_and_empty_run = tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask),
        tf.equal(bill_ratio, 0.0)
    )
    is_medium_cost_level_and_low_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    is_medium_cost_level_and_high_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, medium_mask),
        tf.greater(bill_ratio, 1.3)
    )
    is_high_cost_level_and_empty_run = tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask),
        tf.equal(bill_ratio, 0.0)
    )
    is_high_cost_level_and_low_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    is_high_cost_level_and_high_bill = tf.math.logical_and(
        tf.equal(cost_level_mask, high_mask),
        tf.greater(bill_ratio, 1.3)
    )
    reward = tf.where(
        is_high_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    reward = tf.where(
        is_medium_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    reward = tf.where(
        is_high_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    reward = tf.where(
        is_medium_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    return reward


def reward_shaping_v2(reward, cost_level_mask, bill_ratio, key_to_mask):
    """
    预测得阈值，价值扶持系数，不同cost_level
    Args:
        reward:
        cost_level_mask: 每个样本映射到不同的level:
            high, medium映射到对应 cost_level_info
        bill_ratio: 计费比:
            (0.0, 0.8]: 低估
            0.8-1.2: 正常
            >1.2 高估
            0.0: 空跑
        key_to_mask:  dict
            high: xx
            medium: xx
    Returns:
    """
    high_control_mask = key_to_mask['high_and_control']
    high_treat_mask = key_to_mask['high_and_treat']
    medium_control_mask = key_to_mask['medium_and_control']
    medium_treat_mask = key_to_mask['medium_and_treat']
    is_medium_cost_level_and_empty_run = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.equal(bill_ratio, 0.0)
    )
    is_medium_cost_level_and_low_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    is_medium_cost_level_and_high_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.greater(bill_ratio, 1.3)
    )
    is_high_cost_level_and_empty_run = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, medium_control_mask),
            tf.equal(cost_level_mask, medium_treat_mask)
        ),
        tf.equal(bill_ratio, 0.0)
    )
    is_high_cost_level_and_low_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, high_control_mask),
            tf.equal(cost_level_mask, high_treat_mask)
        ),
        tf.math.logical_and(
            tf.greater(bill_ratio, 0.0),
            tf.less_equal(bill_ratio, 0.8)
        )
    )
    is_high_cost_level_and_high_bill = tf.math.logical_and(
        tf.math.logical_or(
            tf.equal(cost_level_mask, high_control_mask),
            tf.equal(cost_level_mask, high_treat_mask)
        ),
        tf.greater(bill_ratio, 1.3)
    )
    reward = tf.where(
        is_high_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    reward = tf.where(
        is_medium_cost_level_and_empty_run,
        reward * 0.1,
        reward
    )
    reward = tf.where(
        is_high_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    reward = tf.where(
        is_medium_cost_level_and_high_bill,
        reward * 0.8,
        reward
    )
    return reward

def get_pow_w(task_name, is_auto_type, logit_pre, use_trans_learning=False):
    pos_w = tf.constant(1.0)
    if task_name == "convert":
        if use_trans_learning:
            pos_w = tf.constant(12.4)
        else:
            pos_w = tf.where(
                tf.equal(is_auto_type, 0),
                tf.ones_like(logit_pre, tf.float32) * 6.2,
                tf.ones_like(logit_pre, tf.float32) * 17
            )
    elif task_name == "active":
        if use_trans_learning:
            pos_w = tf.constant(3.2)
        else:
            pos_w = tf.where(
                tf.equal(is_auto_type, 0),
                tf.ones_like(logit_pre, tf.float32) * 2.4,
                tf.ones_like(logit_pre, tf.float32) * 3.2
            )
    elif task_name == "cost":
        if use_trans_learning:
            pos_w = tf.constant(4.3)
        else:
            pos_w = tf.where(
                tf.equal(is_auto_type, 0),
                tf.ones_like(logit_pre, tf.float32) * 1.8,
                tf.ones_like(logit_pre, tf.float32) * 3.4
            )
    else:
        raise ValueError("unk task: {}".format(task_name))
    return pos_w
