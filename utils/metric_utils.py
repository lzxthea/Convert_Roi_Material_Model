# encoding:utf-8

import tensorflow as tf

def get_accuracy(pred, label, label_mask, epsilon=1e-5):
    cor_sum = tf.reduce_sum(
        tf.cast(
            tf.equal(pred, label), tf.float32) * label_mask
    )

    cor_avg = cor_sum / tf.maximum(tf.reduce_sum(label_mask), epsilon)

    pad_accuracy = tf.where(
        tf.reduce_sum(label_mask) > 0.0,
        cor_avg,
        1.0
    )
    return pad_accuracy


def get_precision(pred, label, epsilon=1e-5):
    tp_num = tf.reduce_sum(
        pred * label
    )
    tp_fp_num = tf.maximum(tf.reduce_sum(pred), epsilon)
    pad_precision = tf.where(
        tf.reduce_sum(pred) > 0.0,
        tp_num / tp_fp_num,
        1.0
    )
    return pad_precision


def get_recall(pred, label, epsilon=1e-5):
    tp_num = tf.reduce_sum(
        pred * label
    )
    tp_fn_num = tf.maximum(tf.reduce_sum(label), epsilon)
    pad_recall = tf.where(
        tf.reduce_sum(label) > 0.0,
        tp_num / tp_fn_num,
        1.0
    )
    return pad_recall


# In-batch AUC evaluation
def get_AUC(pred, label, label_mask=None):
    assert (len(pred.shape) == 1)
    assert (len(label.shape) == 1)
    label_greater_mat = (tf.reshape(label, [-1, 1]) > label)
    if label_mask is not None:
        assert (len(label_mask.shape) == 1)
        label_mask = (label_mask > 0.0)
        label_greater_mat &= label_mask
        label_greater_mat &= tf.reshape(label_mask, [-1, 1])
    pred_col = tf.reshape(pred, [-1, 1])
    pred_greater_mat = (pred_col > pred)
    pred_equal_mat = tf.equal(pred_col, pred)
    total_pairs = tf.reduce_sum(tf.cast(label_greater_mat, tf.float32))
    greater_sum = tf.reduce_sum(
        tf.cast(pred_greater_mat & label_greater_mat, tf.float32)) + tf.reduce_sum(
        tf.cast(pred_equal_mat & label_greater_mat, tf.float32)) / 2.0
    return tf.where(total_pairs > 0.5, greater_sum / tf.maximum(total_pairs, 1.0), 1.0)


# def wrap_metrics(metrics_dict):
#     """
#     wrap dict for eval_metrics_op
#     Args:
#         metrics_dict:
#
#     Returns:
#
#     """
#     update_dict = dict()
#     for k, v in metrics_dict.items():
#         update_dict[k] = (v, tf.no_op())
#     return update_dict


def wrap_metrics(metrics_dict):
    """
    wrap dict for eval_metrics_op
    Args:
        metrics_dict:

    Returns:

    """
    update_dict = dict()
    for k, v in metrics_dict.items():
        update_dict[k] = tf.reduce_mean(v)
    return update_dict
