# encoding=utf-8
import os
import logging
import tensorflow as tf
import numpy as np
import json

from lagrange_lite.sparse import data, parser, readers
from lagrange_lite.common import metrics
import pickle
if 'TF_CONFIG' not in os.environ:
    _TASK_NAME = 'chief-0'
else:
    tf_config = json.loads(os.environ['TF_CONFIG'])
    cur_task = tf_config['task']
    _TASK_NAME = '%s-%d' % (cur_task['type'], cur_task['index'])

METRICS_TAG_KV = {'task': _TASK_NAME}

# 基于输出的参数构造数据
def create_instance_dataset(
        raw_paths, sparse_keys=[], num_shards=1, shard_id=0, batch_size=1024,
        dense_features={}, shuffle_buffer_size=64 * 1024 * 1024, cycle_length=4,
        block_length=2, num_parallel_maps=None, n_epochs=None,
        num_prefetch=-1,
        is_auto_type=-1
):
    expanded_and_sorted = data.expand_and_sort_paths(raw_paths)
    np.random.shuffle(expanded_and_sorted)

    print("print raw_paths: {}\t files: {}".format(raw_paths, expanded_and_sorted))

    files_to_read = tf.data.Dataset.from_tensor_slices(expanded_and_sorted)
    if num_shards > 1:
        logging.info("Dataset shard is: %d/%d", shard_id, num_shards)
        files_to_read = files_to_read.shard(num_shards, shard_id)

    def parse_fn(serialized):
        features = parser.parse_single_instance(
            serialized,
            sparse_keys=sparse_keys,
            fields={
                'label': tf.io.FixedLenFeature(shape=[7], dtype=tf.float32, default_value=[0.0]*7)},
            lineid_fields={
                'req_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'customer_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'advertiser_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'external_action': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'deep_external_action': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'app_package': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'target_app_package': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'ltr_rank_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                'deep_bid_type': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0]),
                'ad_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=[0])
            },
            dense_fields=dense_features)
        print("All feature's keys that we want to read from dataset: {}".format(features.keys()))
        label = features.pop('label')
        return features, label

    def filter_fn_v2(features, label):
        return tf.reshape(tf.equal(features['fc_dense_auto_ad_type'], is_auto_type), [])
    

    dataset = files_to_read.interleave(
        map_func=lambda one_path: readers.InstanceDataset(
            one_path,
            use_snappy=True,
            has_prefix=True,
            has_sort_id=True,
            has_kafka_dump=False
        ),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=cycle_length,
    ).map(
        parse_fn,
        num_parallel_calls=(tf.data.experimental.AUTOTUNE if num_parallel_maps is None else num_parallel_maps)
    ).shuffle(buffer_size=shuffle_buffer_size)
    print("dataset example:", dataset.take(1))
    if is_auto_type >= 0:
        dataset = dataset.filter(filter_fn_v2)

    dataset = dataset.batch(batch_size)

    if n_epochs is not None and n_epochs > 0:
        dataset = dataset.repeat(n_epochs)

    if num_prefetch > 0:
        dataset = dataset.prefetch(num_prefetch)
    elif num_prefetch == 0:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


class CustomMetricHook(tf.estimator.SessionRunHook):
    ''' Log group of customed metircs for a batch. '''

    def __init__(self, metric_tensors, log_steps=1, ema_decay=0.999):
        for name in metric_tensors:
            tensor = metric_tensors[name]
            if len(tensor.shape.dims) > 0:
                raise ValueError('The metric tensor should be a scalar!')
            if tensor.dtype.base_dtype not in (tf.float32, tf.int32):
                raise ValueError('The dtype of a metric tensor should be either tf.float or tf.int32!')
        if len(metric_tensors) == 0:
            raise ValueError('At least one metric tensor should be offered!')
        assert (log_steps > 0)
        self._metric_tensors = metric_tensors
        self._log_steps = log_steps
        self._step = 0
        self._ema_decay = ema_decay
        self._values = {name: [] for name in metric_tensors}
        self._ema = {name: 0.0 for name in metric_tensors}

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(self._metric_tensors)

    def after_run(self, run_context, run_value):
        metric_values = run_value.results
        for name in metric_values:
            metrics.emit_store(name, float(metric_values[name]), tagkv=METRICS_TAG_KV)
            self._values[name].append(float(metric_values[name]))
            self._ema[name] = self._ema_decay * self._ema[name] + (1.0 - self._ema_decay) * float(metric_values[name])
        self._step += 1
        if self._step % self._log_steps == 0:
            logging.info('Step[{:d}]:'.format(self._step) + ''.join([
                '\t' + name + ': (batch: ' + str(metric_values[name]) +
                ', mean: ' + str(np.mean(self._values[name])) +
                ', ema: ' + str(self._ema[name]) + ')' for name in metric_values]))


if __name__ == "__main__":
    feat_names = {'fc_aid_shadow_cost', 'fc_aid_shadow_cost_last',
                  'fc_dense_external_action_last', 'fc_dense_ad_cid_shadow_num_max_strategy',
                  'fc_dense_ad_cid_shadow_num_max_strategy_last'}
    dense_features = {fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=[0]) for fc in
                      sorted(list(feat_names))}