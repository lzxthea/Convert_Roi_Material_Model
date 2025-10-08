# encoding: utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import numpy as np  
np.set_printoptions(threshold=np.inf)  

from collections import defaultdict

from dataset import create_instance_dataset, CustomMetricHook

from feature_all import DENSE_FEAT_2D_more, SPARSE_FEAT_v2, \
    DENSE_FEAT_1D_v2, NEED_LOG_FEATURE_DICT, \
    RATIO_FEATURE, RATIO_FEATURE_INFO,  \
    DELTA_CAMP_FEAT, DELTA_CAMP_VMID_FEAT, DELTA_VMID_FEAT, DELTA_FEATURE_INFO

from utils.feat_utils import bucket_feats, get_dense_tower, bucket_feats_2d, print_flags, bucket_single_feat_semantic
from dynamic_bucketing_simple import SimpleDynamicBucketing, create_dynamic_bucket_feats_simple

from utils.metric_utils import get_AUC
from utils.loss_utils import get_vars_not_in_scope, get_pow_w, huber_loss, weighted_cross_entropy, cross_entropy, zlin, focal_loss

from lagrange_lite import sparse
from lagrange_lite.tensorflow.train import DeepInsight2Hook, SynchronizedCheckpointSaverHook
from lagrange_lite.common import deep_insight_v2
from lagrange_lite.tensorflow import train
from lagrange_lite.common import JOB_CONTEXT
from lagrange_lite.tensorflow import aop
from datetime import datetime, timedelta
import warnings
# 设置忽略所有警告
warnings.filterwarnings('ignore')

FLAGS = tf.app.flags.FLAGS
# 训练参数
tf.app.flags.DEFINE_string('model_path', './', 'Model save path.')
tf.app.flags.DEFINE_string('train_paths', '', 'Train path')
tf.app.flags.DEFINE_string('test_paths', '', 'Test path')
tf.app.flags.DEFINE_string('last_model_path', '', 'last model path.')
tf.app.flags.DEFINE_integer('is_train', 1, 'train mode')
tf.app.flags.DEFINE_integer('is_eval', 0, 'eval mode')
tf.app.flags.DEFINE_integer('ps_num_embedding_shards', 1, 'ps number')
tf.app.flags.DEFINE_integer('batch_size', 512, 'Testing batch size.')
# tf.app.flags.DEFINE_integer('shuffle_buffer_size', 300000, 'Shuffle buffer size')
tf.app.flags.DEFINE_integer('shuffle_buffer_size', 8192, 'Shuffle buffer size')
tf.app.flags.DEFINE_integer('n_epochs', 1, 'Number of epochs.')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', 300, 'Time interval for saving checkpoints.')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, 'Sumarry steps.')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, 'Log steps.')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 10, 'Checkpoint maximum number.')
tf.app.flags.DEFINE_integer('cycle_length', 8, 'cycle_length')
tf.app.flags.DEFINE_integer('block_length', 2, 'block_length')
tf.app.flags.DEFINE_integer('num_parallel_maps', 256, 'num_parallel_maps')
tf.app.flags.DEFINE_bool('batch_reload', True, '天级别追新的时候是否reload模型')
tf.app.flags.DEFINE_float('deep_instance_sample_ratio', 0.1, 'deep_instance_sample_ratio')
tf.app.flags.DEFINE_float('huber_delta', 1.0, 'huber_delta')

# 模型参数
tf.app.flags.DEFINE_integer('seed', 9431, 'random seed')
tf.app.flags.DEFINE_string('dnn_hidden_dims_common', '[512,128,64]', 'DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('small_dnn_hidden_dims_common', '[128,64]', 'small DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('shared_dnn_hidden_dims_common', '[128,64]', 'SHARED DNN hidden dimensionality list in common.')
tf.app.flags.DEFINE_string('bias_hidden_dims_common', '[16, 4]', 'bias hidden dimensionality list 2 in common.')
# tf.app.flags.DEFINE_string('need_task', 'convert,cost,convert_per_hour_label,cost_per_hour_label,convert_per_ad_hour_label,cost_per_ad_hour_label', '是否选择某个任务')
tf.app.flags.DEFINE_integer('sparse_emb_dim', '4', '稀疏特征的emb_size')
tf.app.flags.DEFINE_integer('embedding_dim', '4', '参数embedding后的维度')
tf.app.flags.DEFINE_float('learning_rate', 0.007, 'Learning rate.')
tf.app.flags.DEFINE_float('ema_decay', 0.99, 'EMA decay for hooks.')
tf.app.flags.DEFINE_float('dropout_prob', 0.0, '是否使用dropout')
tf.app.flags.DEFINE_float('cost_thresh', 100.0, 'cost_level的切换阈值.')
tf.app.flags.DEFINE_bool('separate_embedding', True, '是否使用隔离embedding')
tf.app.flags.DEFINE_bool('separate_dense_embedding', False, '是否使用隔离dense embedding')
tf.app.flags.DEFINE_bool('seq_pooling', False, '是否使用序列感知建模')
tf.app.flags.DEFINE_bool('transfer_learning', True, '是否迁移')
tf.app.flags.DEFINE_bool('use_log_feature', True, 'use log feature.')
tf.app.flags.DEFINE_bool('use_din', True, 'use_din')
tf.app.flags.DEFINE_integer('auto_type_to_keep', -1, '是否过滤')
tf.app.flags.DEFINE_bool('enable_ad_avg_uplift', True, 'enable_ad_avg_uplift')
tf.app.flags.DEFINE_bool('use_dynamic_bucketing', True, '是否使用动态分桶')
tf.app.flags.DEFINE_integer('dynamic_bucketing_batches', 1000, '动态分桶统计的batch数量')
tf.app.flags.DEFINE_string('dynamic_bucketing_path', './bucket_points.json', '动态分桶点保存路径')
tf.app.flags.DEFINE_float('label_cap_value_lb', 0.0, 'label_cap_value_lb')
tf.app.flags.DEFINE_float('label_cap_value_ub', 300.0, 'label_cap_value_ub')
tf.app.flags.DEFINE_string('loss_type', 'weighted_cross_entropy, focal_loss', 'loss_type')
tf.app.flags.DEFINE_float("temp", 10, "")
tf.app.flags.DEFINE_integer('adcnt_embedding_dim', 4, 'ad_cnt embedding后的维度')
tf.app.flags.DEFINE_integer('is_offline', 0, 'eval mode')
tf.app.flags.DEFINE_float('cross_entropy_weight', 2.0, 'cross_entropy_weight')
print_flags(FLAGS)
print('is_offline:', FLAGS.is_offline)
print('loss_type:', FLAGS.loss_type)
print('shuffle_buffer_size:', FLAGS.shuffle_buffer_size)
print('cross_entropy_weight:', FLAGS.cross_entropy_weight)
print('use_dynamic_bucketing:', FLAGS.use_dynamic_bucketing)
print('dynamic_bucketing_batches:', FLAGS.dynamic_bucketing_batches)

# --------------- init path ------------------------
# 处理离线训练数据
def get_offline_train_eval_path(train_paths):
    train_path_list = train_paths.split('|')
    new_train_path, new_eval_path = [], []
    for train_path in train_path_list:
        base_path_length = len(train_path) - len(train_path.split('/')[-1]) - len(train_path.split('/')[-2]) - 1
        base_path = train_path[:base_path_length]
        rest = train_path[base_path_length:]
        if rest[-2:] == '/*':
            dates = rest.rstrip('/*').lstrip('{').rstrip('}').split(',')
        else:
            dates = rest.rstrip('/part-*').lstrip('{').rstrip('}').split(',')
        eval_date = dates[-1]
        if len(dates) == 1:
            train_dates = [(datetime.strptime(eval_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')]
        else:
            train_dates = dates[:-1]
        if rest[-2:] == '/*':
            train_path, eval_path = base_path + '{' + ','.join(train_dates) + '}/*', base_path + '{' + eval_date + '}/*'
        else:
            train_path, eval_path = base_path + '{' + ','.join(train_dates) + '}/part-*', base_path + '{' + eval_date + '}' + '/part-*'
        new_train_path.append(train_path)
        new_eval_path.append(eval_path)
    new_train_path = '|'.join(new_train_path)
    new_eval_path = '|'.join(new_eval_path)
    return new_train_path, new_eval_path

if FLAGS.is_offline:
    FLAGS.train_paths = FLAGS.train_paths
    FLAGS.test_paths = FLAGS.test_paths

print('[test] train_paths: ', FLAGS.train_paths)
print('[test] test_path: ', FLAGS.test_paths)
print('[test] is_offline: ', FLAGS.is_offline)

deep_insight_v2.reset(deep_insight_sample_ratio=FLAGS.deep_instance_sample_ratio)

SHARED_DNN_DIMS_COMMON = eval(FLAGS.shared_dnn_hidden_dims_common)
DNN_DIMS_COMMON = eval(FLAGS.dnn_hidden_dims_common) + [1]
SMALL_DNN_DIMS_COMMON = eval(FLAGS.small_dnn_hidden_dims_common) + [1]
BIAS_DIMS_COMMON = eval(FLAGS.bias_hidden_dims_common) + [1]

EME_DIM = FLAGS.embedding_dim # dense特征的embedding长度
# lzx 修改
TASK_NAME = ["convert","roi1"]
TASK_NAMES_DICT = {"convert":0,"roi1": 2} # label为roi1

# TASK_NAME = ["pid_model"]
# TASK_NAMES_DICT = {"pid_model": 2} # label为roi1

# NEED_TASK = FLAGS.need_task.split(",")
# TASK_NAME = NEED_TASK
# TASK_NAMES_DICT = {"convert": 0,"cost": 1,"convert_per_hour_label": 2,"cost_per_hour_label": 3,"convert_per_ad_hour_label": 4,"cost_per_ad_hour_label": 5}

def supervised_model_fn(model, features, labels, mode, params, config):
    logging_hook_di = dict()
    tf.logging.info("features keys:{}".format(features.keys()))

    sparse.feature.FeatureSlot.set_default_bias_initializer(tf.zeros_initializer())
    sparse.feature.FeatureSlot.set_default_vec_initializer(tf.random_uniform_initializer(-0.0078125, 0.0078125))
    sparse.feature.FeatureSlot.set_default_bias_optimizer(
        sparse.optimizer.SparseFtrlOptimizer(learning_rate=FLAGS.learning_rate,
                                             l1_regularization_strength=1.2,
                                             l2_regularization_strength=0.01))
    sparse.feature.FeatureSlot.set_default_vec_optimizer(
        sparse.optimizer.SparseAdagradOptimizer(learning_rate=FLAGS.learning_rate))

    state_embeddings_names = []
    camp_vmid_feat_hourly = list()

    # embeding维度
    sparse_emb_dim = FLAGS.sparse_emb_dim
    
    # 字典收集不同任务下的embedding列表
    state_embeddings_mt = defaultdict(list)

    # 特征字典，key为特征名称，value为分桶边界值
    sparse_features = SPARSE_FEAT_v2
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    # key为需要log化的特征名，value为log化后的分桶边界值
    need_log_feature = dict()
    if FLAGS.use_log_feature:
        need_log_feature = NEED_LOG_FEATURE_DICT

    # 三元组（'name_1', 'name_2', 'fc_name'），'fc_name'的特征 = 'name_1'特征 / 'name_2'特征
    ratio_list = RATIO_FEATURE
    ratio_feat_info = RATIO_FEATURE_INFO
    delta_camp = DELTA_CAMP_FEAT
    delta_camp_vmid = DELTA_CAMP_VMID_FEAT
    delta_vmid = DELTA_VMID_FEAT
    delta_feat_info = DELTA_FEATURE_INFO

    # 稀疏特征
    # for feat_name, slot_id_hash_size in sorted(sparse_features.items()):
    #     slot_id, hash_size = slot_id_hash_size
    #     fs = model.add_feature_slot(slot_id, hash_size)
    #     if slot_id < 1024:
    #         fc = model.add_feature_column_v1(fs)
    #     else:  # fid v2
    #         fc = model.add_feature_column_v2(feat_name, fs)
    #     if FLAGS.separate_embedding:
    #         for t in TASK_NAME:
    #             state_embeddings_mt[t].append(fc.add_vector(sparse_emb_dim))
    #     else:
    #         common_emb = fc.add_vector(sparse_emb_dim)
    #         for t in TASK_NAME:
    #             state_embeddings_mt[t].append(common_emb)
    #     state_embeddings_names.append(feat_name)
    for feat_name, slot_id_hash_size in sorted(sparse_features.items()):
        slot_id, hash_size = slot_id_hash_size
        fs = model.add_feature_slot(slot_id, hash_size)
        fc = model.add_feature_column_v1(fs)
        for t in TASK_NAME:
            state_embeddings_mt[t].append(fc.add_vector(sparse_emb_dim))
        state_embeddings_names.append(feat_name)
    
    print("len sparse_features:", len(state_embeddings_names))

    # dense特征、campaign特征
    with tf.variable_scope("model_cur", reuse=tf.AUTO_REUSE,
                           partitioner=tf.fixed_size_partitioner(FLAGS.ps_num_embedding_shards, axis=0)):
        # 1d特征处理 - 支持动态分桶
        if FLAGS.use_dynamic_bucketing:
            # 初始化动态分桶器
            dynamic_bucketing = SimpleDynamicBucketing(
                num_batches_for_stats=FLAGS.dynamic_bucketing_batches,
                save_path=FLAGS.dynamic_bucketing_path
            )
            
            # 尝试加载已保存的分桶点
            dynamic_bucketing.load_bucket_points()
            
            # 收集特征统计信息（仅在训练模式下）
            if mode == tf.estimator.ModeKeys.TRAIN and not dynamic_bucketing.is_initialized:
                dynamic_bucketing.collect_feature_stats(features, list(dense_features_1d.keys()))
            
            # 使用动态分桶处理1D特征
            dense_features_1d_embeddings, dense_features_1d_names = create_dynamic_bucket_feats_simple(
                features, dense_features_1d, dynamic_bucketing,
                log_dict=need_log_feature, need_log1p=False, dim=EME_DIM,
                all_feat_suffix="common"
            )
        else:
            # 使用原始静态分桶
            dense_features_1d_embeddings, dense_features_1d_names, _ = bucket_feats(
                features, dense_features_1d, log_dict=need_log_feature, need_log1p=False, dim=EME_DIM,
                all_feat_suffix="common"
            )
        
        state_embeddings_names += dense_features_1d_names
        for t in TASK_NAME:
            state_embeddings_mt[t] += dense_features_1d_embeddings
        print("len dense_features_1d_names:", len(dense_features_1d_names))

        # 2d特征处理
        dense_features_2d_embeddings, dense_features_2d_names, all_emb_size, \
        campaign_embeddings, campaign_vmid_embedding, vmid_emb_size = bucket_feats_2d(
            features, dense_features_2d, log_dict=need_log_feature, need_log1p=False, dim=EME_DIM, need_reduce="max",
            all_feat_suffix="common", seq_pooling=FLAGS.seq_pooling,
            ratio_feat_list=ratio_list, ratio_feat=ratio_feat_info,
            delta_camp=delta_camp,
            delta_camp_vmid=delta_camp_vmid, delta_vmid=delta_vmid, delta_feat=delta_feat_info,
        )

        for t in TASK_NAME:
            state_embeddings_mt[t] += dense_features_2d_embeddings

        state_embeddings_names += dense_features_2d_names
        print("len dense_features_2d_names:", len(dense_features_2d_names))
        print("len state_embeddings_names(eqal_sparse+dense1d+dense2d):", len(state_embeddings_names))

    # 打印tensorboard
    for t in TASK_NAME:
        assert len(state_embeddings_mt[t]) == len(
            state_embeddings_names), "len_state_embeddings_mt_{}: {} \t len_state_embeddings_names: {}".format(
            t, len(state_embeddings_mt[t]), len(state_embeddings_names)
        )
        is_all_fea_same = dict()
        for idx in range(len(state_embeddings_names)):
            # 记录各个特征，每个batch内是否相同
            feat_mean = tf.reduce_sum(state_embeddings_mt[t][idx], axis = 1)
            tf.summary.histogram(
                'sum_{}_{}'.format(t, state_embeddings_names[idx]), feat_mean
            )
            tf.summary.histogram(
                'feat_{}_{}'.format(t, state_embeddings_names[idx]), state_embeddings_mt[t][idx]
            )
    
        print('feat_mean.shape:', feat_mean.shape)

    # 对之前分任务的特征做汇总整合
    state_embedding_mt = dict()
    pred_mt = dict()

    for t in TASK_NAME:
        state_embedding_mt[t] = tf.concat(state_embeddings_mt[t], axis=1)
        print('state_embedding_mt[t].shape', state_embedding_mt[t].shape)

    with tf.variable_scope("model_cur", reuse=tf.AUTO_REUSE,
                           partitioner=tf.fixed_size_partitioner(FLAGS.ps_num_embedding_shards, axis=0)
                           ):
        # [batch, 1]
        logit_mt = list()
        is_train = (mode == tf.estimator.ModeKeys.TRAIN)
        for task in TASK_NAME:
            state_tensor = state_embedding_mt[task]
            name = "all"
            print("state_tensor.shape: ", state_tensor.shape) # (?, 1876)

            # 打印emb
            # pred_mt['input_tensor'] = state_tensor

            state_embedding_logit = get_dense_tower(
                # DNN_DIMS_COMMON,
                [512, 256, 64, 1],
                state_tensor,
                "task_{}_score_{}".format(task, name),
                is_train=is_train,
                dropout_prob=FLAGS.dropout_prob
            )

            state_embedding_logit = tf.reshape(state_embedding_logit, [-1])
            print("state_embedding_logit.shape: ", state_embedding_logit.shape)
            logit_mt.append(state_embedding_logit)
            pred_mt[task] = state_embedding_logit
    
    model.freeze_slots(features)

    # 添加模型参数
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_dict = {}
        for task in TASK_NAME:
            predictions_dict[task] = pred_mt[task]
        # predictions_dict['pid_model'] = pred_mt['pid_model']
                   
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)
    else:
        loss_merge = 0
        eval_tensors = dict()
        score_mt_dict = dict()
        label_mt_dict = dict()
        value_mt_dict = dict()
        logit_post_mt_dict = dict()
        
        label_mt = {}
        loss_mt = {}
        metrics_dict = dict()

        for task, idx in TASK_NAMES_DICT.items():
            label_mt[task] = labels[:, idx] # label为roi

        for task in TASK_NAME:
            task_score = tf.reshape(pred_mt[task], [-1])
            raw_label = tf.reshape(label_mt[task], [-1])

            # 分类问题
            task_label = tf.clip_by_value(raw_label, FLAGS.label_cap_value_lb, FLAGS.label_cap_value_ub)
            if task == "convert": # convert>0
                binary_label = tf.cast(task_label>0, tf.float32)
            else: # roi>=1
                binary_label = tf.cast(task_label>=1, tf.float32) # 值域0-1
            # binary_pred = tf.nn.sigmoid(task_score - 1) # 值域0.5-1 预期: roi>=1时，pred>0.5;roi<1时，pred<0.5
            # binary_pred = tf.nn.sigmoid(task_score) 
            binary_pred = task_score # pred_mt[task] = tf.nn.sigmoid(state_embedding_logit) 转化过

            # 回归问题
            pred_score = tf.nn.relu(task_score)
            reg_pred = tf.where(pred_score <= 100, 
                                pred_score,               # 保持原值
                                100 + 1.0 * tf.math.log(pred_score - 100 + 1.0)  # log平滑
                                )
            reg_label = tf.where(raw_label <= 100,  
                                raw_label,               # 保持原值
                                100 + 1.0 * tf.math.log(raw_label - 100 + 1.0)  # log平滑
                                )

            # loss
            loss_list = FLAGS.loss_type.split(",")
            # 样本加权
            reg_weight = tf.where(reg_label <= 1,
                                tf.ones_like(reg_label, dtype=tf.float32),  # label<=1时权重为1
                                reg_label
                                )
            
            # label & pred分布
            tf.summary.histogram("binary_label", binary_label)
            tf.summary.histogram("binary_pred", binary_pred)
            tf.summary.histogram("reg_pred", reg_pred)
            tf.summary.histogram("reg_label", reg_label)
            tf.summary.histogram("reg_weight", reg_weight)
            loss = 0
            if 'mae' in loss_list:
                loss_pre = tf.abs(reg_label - reg_pred)
                mae_loss = tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:mae")
            if 'huber' in loss_list:
                loss_pre = huber_loss(reg_label, reg_pred, logging_hook_di, FLAGS.huber_delta)
                # huber_loss_mean = tf.reduce_mean(reg_weight * loss_pre)
                huber_loss_mean = tf.reduce_mean(loss_pre)
                loss += huber_loss_mean
                tf.logging.info("[debug] loss:huber_loss")
            if 'weighted_cross_entropy' in loss_list:
                loss_pre = weighted_cross_entropy(binary_label, binary_pred, FLAGS.cross_entropy_weight, logging_hook_di)
                loss += tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:weighted_cross_entropy")
            if 'focal_loss' in loss_list:
                loss_pre = focal_loss(binary_label, binary_pred)
                loss += tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss: focal_loss")
            if 'cross_entropy' in loss_list:
                loss_pre = cross_entropy(binary_label, binary_pred, logging_hook_di)
                ce_loss = tf.reduce_mean(loss_pre)
                loss += ce_loss
                tf.logging.info("[debug] loss:cross_entropy")
            if 'zlin' in loss_list:
                loss_pre = zlin(raw_label, task_score, logging_hook_di)
                loss += tf.reduce_mean(loss_pre)
                tf.logging.info("[debug] loss:zlin")
                
            # logging_hook_di['label_2'] = tf.reshape(labels[:, 2], [-1]) # roi1的label
            loss_mt[task] = loss
            loss_merge += loss
            with tf.name_scope("Loss"):
                tf.summary.scalar(
                    "loss_mean_auto_{}".format(task), loss
                )

            with tf.name_scope("Metric"):
                # metrics_dict['loss'] = loss
                #metrics_dict['mae_{}'.format(task)] = get_mae(task_score, task_label)
                #metrics_dict['mse_{}'.format(task)] = get_mse(task_score, task_label)
                metrics_dict['label_{}'.format(task)] = tf.reduce_mean(task_label)
                metrics_dict['score_{}'.format(task)] = tf.reduce_mean(task_score)

            eval_tensors.update(metrics_dict)
            

        req_time_tensor = tf.fill(tf.shape(logit_mt[0]), tf.timestamp(name=None))
        mock_user = tf.fill(tf.shape(logit_mt[0]), 0)

        eval_metric_prediction = CustomMetricHook(
            eval_tensors, log_steps=FLAGS.save_summary_steps, ema_decay=FLAGS.ema_decay)
        for k, v in metrics_dict.items():
            tf.summary.scalar(
                k, tf.reduce_mean(v)
            )
        scaffold = tf.compat.v1.train.Scaffold()
        if mode == tf.estimator.ModeKeys.EVAL:
            loss_eval = loss_merge
            eval_summary_hook = tf.estimator.SummarySaverHook(
                save_steps=20,
                output_dir=os.path.join(JOB_CONTEXT.summary_dir, 'eval'),
                scaffold=scaffold
            )
            save_scores_tensor_dict = dict()
            save_labels_tensor_dict = dict()
            save_task_names = TASK_NAME

            logging_hook_eval = dict()
            # logging_hook_eval['fc_ecp_product_id'] = tf.reshape(features['fc_ecp_product_id'], [-1])
            for task in TASK_NAME:
                logging_hook_eval[task] = tf.reshape(pred_mt[task], [-1])
            logging_hook_eval['eval_label'] = tf.reshape(labels[:, 2], [-1])
            # logging_hook_eval['input_tensor'] = pred_mt['input_tensor']
            logging_hook_dump = tf.train.LoggingTensorHook(tensors=logging_hook_eval, every_n_iter=1)

            for task_name in save_task_names:
                # label统计
                save_labels_tensor_dict[task_name] = tf.reshape(label_mt[task_name], [-1])
                # predict统计
                save_scores_tensor_dict[task_name] = tf.reshape(pred_mt[task_name], [-1])

                # cap
                # label统计
                save_labels_tensor_dict[task_name+"_cap"] = tf.reshape(tf.clip_by_value(label_mt[task_name], 0, 300), [-1])
                # predict统计
                save_scores_tensor_dict[task_name+"_cap"] = tf.reshape(pred_mt[task_name], [-1])
            
            # 写入deepinsight
            di2_multihead_hook = train.DeepInsight2MultiHeadHook(
                tf.reshape(mock_user, [-1]),
                tf.reshape(req_time_tensor, [-1]),
                score_tensor_dict=save_scores_tensor_dict,
                label_tensor_dict=save_labels_tensor_dict,
                extra_tensors={
                    'dataset': tf.fill(tf.shape(logit_mt[0]), 'eval'),
                    'req_id': tf.reshape(features['req_id'], [-1]),
                    'customer_id': tf.reshape(features['customer_id'], [-1]),
                    'advertiser_id': tf.reshape(features['advertiser_id'], [-1]),
                    # 'product_id': tf.reshape(features['product_id'], [-1]),
                    'external_action': tf.reshape(features['external_action'], [-1]),
                    'deep_external_action': tf.reshape(features['deep_external_action'], [-1]),
                    'deep_bid_type': tf.reshape(features['deep_bid_type'], [-1]),
                    'app_package': tf.reshape(features['app_package'], [-1]),
                    'target_app_package': tf.reshape(features['target_app_package'], [-1]),
                    'fc_real_cost_72h_vmid_all': tf.reshape(features['fc_real_cost_72h_vmid_all'], [-1]),
                    'fc_pvr_72h_vmid_all': tf.reshape(features['fc_pvr_72h_vmid_all'], [-1]),
                    'convert_label': tf.reshape(labels[:, 0], [-1]),
                    'gmv_label': tf.reshape(labels[:, 1], [-1]),
                    'roi1_label': tf.reshape(labels[:, 2], [-1]),
                    'cost_label': tf.reshape(labels[:, 3], [-1]),
                    'ctr__label': tf.reshape(labels[:, 4], [-1]),
                    'cvr_label': tf.reshape(labels[:, 5], [-1]),
                    'p_date': tf.reshape(labels[:, 6], [-1]),

                },
                neg_sample_rate=1.0
            )
            update_metric_tensors = dict()
            for k, v in eval_tensors.items():
                update_metric_tensors[k] = tf.metrics.mean(v)

            for task_name in TASK_NAME:

                # 分类问题
                task_label = tf.clip_by_value(save_labels_tensor_dict[task_name], FLAGS.label_cap_value_lb, FLAGS.label_cap_value_ub)
                if task == "convert": # convert>0
                    binary_label = tf.cast(task_label>0, tf.float32)
                else: # roi>=1
                    binary_label = tf.cast(task_label>=1, tf.float32) # 值域0-1
                # binary_pred = tf.nn.sigmoid(task_score - 1) # 回归问题里的binary值域0.5-1 预期: roi>=1时，pred>0.5;roi<1时，pred<0.5
                # binary_pred = tf.nn.sigmoid(task_score)
                binary_pred = task_score # pred_mt[task] = tf.nn.sigmoid(state_embedding_logit) 转化过

                # 回归问题
                reg_pred = tf.where(task_score <= 100, 
                                    task_score,               # 保持原值
                                    100 + 1.0 * tf.math.log(task_score - 100 + 1.0)  # log平滑
                                    )
                reg_label = tf.where(raw_label <= 100,  
                                    raw_label,               # 保持原值
                                    100 + 1.0 * tf.math.log(raw_label - 100 + 1.0)  # log平滑
                                    )

                # 不同阈值的acc
                thresholds = [i * 0.1 for i in range(1, 10)]  # 生成[0.1, 0.2, ..., 0.9]
                for thres in thresholds:
                    metric_name = f"acc_{task_name}_threshold_{thres:.1f}"
                    op_name = f"acc_op_{task_name}_threshold_{thres:.1f}"

                    update_metric_tensors[metric_name] = tf.metrics.accuracy(
                        # labels=save_labels_tensor_dict[task_name],
                        labels=binary_label,
                        predictions=tf.cast(binary_pred >= thres, tf.float32),
                        name=op_name)

                update_metric_tensors["FP_{}".format(task_name)] = tf.metrics.false_positives(
                    # labels=save_labels_tensor_dict[task_name],
                    labels=binary_label,
                    predictions=tf.cast(binary_pred >= 0.5, tf.float32),
                    weights=None, 
                    metrics_collections=None,
                    updates_collections=None, 
                    name='fp_op_{}'.format(task_name))
                
                update_metric_tensors["FN_{}".format(task_name)] = tf.metrics.false_negatives(
                    # labels=save_labels_tensor_dict[task_name],
                    labels=binary_label,
                    predictions=tf.cast(binary_pred >= 0.5, tf.float32),
                    weights=None, 
                    metrics_collections=None,
                    updates_collections=None, 
                    name='fn_op_{}'.format(task_name))

                update_metric_tensors["TP_{}".format(task_name)] = tf.metrics.true_positives(
                    # labels=save_labels_tensor_dict[task_name],
                    labels=binary_label,
                    predictions=tf.cast(binary_pred >= 0.5, tf.float32),
                    weights=None, 
                    metrics_collections=None,
                    updates_collections=None, 
                    name='tp_op_{}'.format(task_name))
                
                update_metric_tensors["TN_{}".format(task_name)] = tf.metrics.true_negatives(
                    # labels=save_labels_tensor_dict[task_name],
                    labels=binary_label,
                    predictions=tf.cast(binary_pred >= 0.5, tf.float32),
                    weights=None, 
                    metrics_collections=None,
                    updates_collections=None, 
                    name='tn_op_{}'.format(task_name))
                
                update_metric_tensors["mse_{}".format(task_name)] = tf.compat.v1.metrics.mean_squared_error(
                        labels=reg_label,
                        predictions=reg_pred,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=f'mse_op_{task_name}'
                    )
                
                update_metric_tensors["auc_{}".format(task_name)] = tf.metrics.auc(
                    # labels=save_labels_tensor_dict[task_name],
                    labels=binary_label,
                    predictions=binary_pred,
                    weights=None, num_thresholds=200, metrics_collections=None,
                    updates_collections=None, curve='ROC',
                    summation_method='trapezoidal', thresholds=None,
                    name='auc_op_{}'.format(task_name))

                update_metric_tensors["recall_{}".format(task_name)] = tf.metrics.auc(
                    # labels=save_labels_tensor_dict[task_name],
                    labels=binary_label,
                    predictions=binary_pred,
                    thresholds=[0.08, 0.1, 0.2, 0.5],
                    weights=None,
                    metrics_collections=None, updates_collections=None,
                    name='recall_op_' + task_name
                )

                # 修正：计算召回率，使用正确的函数 tf.metrics.recall
                update_metric_tensors["recall_for_{}".format(task_name)] = tf.metrics.recall(
                    labels=binary_label,  # 真实标签（0或1）
                    predictions=tf.cast(binary_pred >= 0.5, tf.float32),  # 预测结果（需转为0/1，这里以0.5为阈值）
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name='recall_op_' + task_name
                )

                update_metric_tensors["reg_pred_mean_{}".format(task_name)] = tf.metrics.mean(
                    values=reg_pred,
                    weights=None, metrics_collections=None,
                    updates_collections=None, name='reg_predict_mean_op_' + task_name)

                update_metric_tensors["binary_pred_mean_{}".format(task_name)] = tf.metrics.mean(
                    values=binary_pred,
                    weights=None, metrics_collections=None,
                    updates_collections=None, name='binary_predict_mean_op_' + task_name)
                

                update_metric_tensors["reg_label_mean_{}".format(task_name)] = tf.metrics.mean(
                    # values=save_labels_tensor_dict[task_name],
                    values=tf.cast(reg_label, tf.float32), # roi >1为正样本
                    weights=None, metrics_collections=None,
                    updates_collections=None, name='reg_label_mean_op_' + task_name)


                update_metric_tensors["binary_label_mean_{}".format(task_name)] = tf.metrics.mean(
                    # values=save_labels_tensor_dict[task_name],
                    values=binary_label, # roi >1为正样本
                    weights=None, metrics_collections=None,
                    updates_collections=None, name='binary_label_mean_op_' + task_name)
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_eval,
                                              eval_metric_ops=update_metric_tensors,
                                              evaluation_hooks=[eval_metric_prediction, eval_summary_hook, di2_multihead_hook, logging_hook_dump])
        else:
            save_scores_tensor_dict = dict()
            save_labels_tensor_dict = dict()
            save_task_names = TASK_NAME
            for task_name in save_task_names:
                # label统计
                save_labels_tensor_dict[task_name] = tf.reshape(label_mt[task_name], [-1])
                # predict统计
                save_scores_tensor_dict[task_name] = tf.reshape(pred_mt[task_name], [-1])

                # cap
                # label统计
                save_labels_tensor_dict[task_name+"_cap"] = tf.reshape(tf.clip_by_value(label_mt[task_name], 0, 300), [-1])
                # predict统计
                save_scores_tensor_dict[task_name+"_cap"] = tf.reshape(pred_mt[task_name], [-1])

            # 写入deepinsight
            di2_multihead_hook = train.DeepInsight2MultiHeadHook(
                tf.reshape(mock_user, [-1]),
                tf.reshape(req_time_tensor, [-1]),
                score_tensor_dict=save_scores_tensor_dict,
                label_tensor_dict=save_labels_tensor_dict,
                extra_tensors={
                    'dataset': tf.fill(tf.shape(logit_mt[0]), 'train'),
                    # 'req_id': tf.reshape(features['req_id'], [-1]),
                    # 'customer_id': tf.reshape(features['customer_id'], [-1]),
                    # 'advertiser_id': tf.reshape(features['advertiser_id'], [-1]),
                    # 'product_id': tf.reshape(features['product_id'], [-1]),
                    'external_action': tf.reshape(features['external_action'], [-1]),
                    # 'deep_external_action': tf.reshape(features['deep_external_action'], [-1]),
                    # 'deep_bid_type': tf.reshape(features['deep_bid_type'], [-1]),
                    # 'app_package': tf.reshape(features['app_package'], [-1]),
                    # 'target_app_package': tf.reshape(features['target_app_package'], [-1]),
                    # 'fc_real_cost_72h_vmid_all': tf.reshape(features['fc_real_cost_72h_vmid_all'], [-1]),
                    # 'fc_pvr_72h_vmid_all': tf.reshape(features['fc_pvr_72h_vmid_all'], [-1]),
                    'convert_label': tf.reshape(labels[:, 0], [-1]),
                    'gmv_label': tf.reshape(labels[:, 1], [-1]),
                    'roi1_label': tf.reshape(labels[:, 2], [-1]),
                    'cost_label': tf.reshape(labels[:, 3], [-1]),
                    'ctr__label': tf.reshape(labels[:, 4], [-1]),
                    'cvr_label': tf.reshape(labels[:, 5], [-1]),
                    'p_date': tf.reshape(labels[:, 6], [-1]),
                },
                neg_sample_rate=1.0
            )
            main_opt = tf.compat.v1.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
            main_vars = get_vars_not_in_scope("model_bias")
            main_grads_and_vars = main_opt.compute_gradients(loss=loss_merge, var_list=main_vars)
            for v, (grad, val) in zip(main_vars, main_grads_and_vars):
                if grad is not None:
                    tf.summary.histogram("grad_{}".format(v), grad)
            train_op = main_opt.apply_gradients(main_grads_and_vars, global_step=tf.compat.v1.train.get_global_step())
            
            logging_hook = tf.train.LoggingTensorHook(tensors=logging_hook_di, every_n_iter=1)
            synced_saver_hook = SynchronizedCheckpointSaverHook(config, scaffold)
            training_chief_hooks = [tf.train.ProfilerHook(save_steps=1000)]
            
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss_merge,
                                              train_op=train_op,
                                              scaffold=scaffold,
                                              training_hooks=[synced_saver_hook, di2_multihead_hook, logging_hook],
                                              training_chief_hooks=training_chief_hooks)


def get_estimator():
    checkpoint_dir = os.path.join(FLAGS.model_path, 'checkpoints')
    if FLAGS.last_model_path and not tf.train.latest_checkpoint(checkpoint_dir):
        if FLAGS.batch_reload:
            warmup_dir = os.path.join(FLAGS.last_model_path, 'checkpoints')
            print("input last_model_path: {} \t do reload!".format(checkpoint_dir))
        else:
            warmup_dir = None
            print("input last_model_path: {} \t but do not reload!".format(checkpoint_dir))
    else:
        warmup_dir = None
    config = tf.estimator.RunConfig(save_summary_steps=FLAGS.save_summary_steps,
                                    log_step_count_steps=FLAGS.log_step_count_steps,
                                    save_checkpoints_secs=FLAGS.save_checkpoints_secs,
                                    keep_checkpoint_max=FLAGS.keep_checkpoint_max,
                                    model_dir=checkpoint_dir,
                                    tf_random_seed=FLAGS.seed
                                    )
    return sparse.estimator.SparseEstimator(
        model_fn=supervised_model_fn,
        num_embedding_shards=None if FLAGS.is_train == 1 else FLAGS.ps_num_embedding_shards,
        config=config,
        warm_start_from=warmup_dir), config


def train_and_evaluate():
    model, run_config = get_estimator()

    sparse_features = SPARSE_FEAT_v2
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    dense_features_1d_real_2d = [k for k in dense_features_1d.keys() if k in dense_features_2d]
    dense_features_1d_real_1d = [k for k in dense_features_1d.keys() if k not in dense_features_2d]
    # dense_features_1d_real_1d = [k for k in dense_features_1d.keys()]

    dense_features = {fc: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=[0]) for fc in
                      sorted(dense_features_1d_real_1d)}

    dense_features.update({fc: tf.io.FixedLenFeature(shape=[20], dtype=tf.int64, default_value=[0] * 20) for fc in
                           sorted(list(dense_features_2d.keys())) + dense_features_1d_real_2d})

    num_shards = run_config.num_worker_replicas
    shard_id = run_config.task_id + (0 if run_config.is_chief else 1)

    if FLAGS.is_train == 1:
        print('Training start!')
        model.train(
            input_fn=lambda: create_instance_dataset(FLAGS.train_paths,
                                                     sparse_keys=list(sparse_features.keys()),
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     batch_size=FLAGS.batch_size,
                                                     dense_features=dense_features,
                                                     shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                                     n_epochs=1,
                                                     cycle_length=FLAGS.cycle_length,
                                                     block_length=FLAGS.block_length,
                                                     num_parallel_maps=FLAGS.num_parallel_maps,
                                                     is_auto_type=FLAGS.auto_type_to_keep
                                                     ),
        )
        print('Training finishes!')

    if FLAGS.is_eval == 1:
        print('Evaluation starts!')
        eval_metrics = model.evaluate(
            input_fn=lambda: create_instance_dataset(FLAGS.test_paths,
                                                     sparse_keys=list(sparse_features.keys()),
                                                     num_shards=num_shards,
                                                     shard_id=shard_id,
                                                     batch_size=FLAGS.batch_size,
                                                     dense_features=dense_features,
                                                     shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                                     n_epochs=FLAGS.n_epochs,
                                                     cycle_length=FLAGS.cycle_length,
                                                     block_length=FLAGS.block_length,
                                                     num_parallel_maps=FLAGS.num_parallel_maps,
                                                     is_auto_type=FLAGS.auto_type_to_keep
                                                     )
            , steps=None
        )
        print('eval metric_ops = {}'.format(eval_metrics))
        print('Evaluation finishes!')

    return model, run_config


def serving_input_receiver_fn():
    dense_features_1d = DENSE_FEAT_1D_v2
    dense_features_2d = DENSE_FEAT_2D_more

    features_next = {
        'fids_indices': tf.placeholder(tf.int64, shape=[None], name='fids_indices'),
        'fids_values': tf.placeholder(tf.int64, shape=[None], name='fids_values'),
        'fids_dense_shape': tf.placeholder(tf.int64, shape=[None], name='fids_dense_shape')
    }
    dense_features_1d_real_2d = [k for k in dense_features_1d.keys() if k in DENSE_FEAT_2D_more]
    dense_features_1d_real_1d = [k for k in dense_features_1d.keys() if k not in DENSE_FEAT_2D_more]
    # dense_features_1d_real_1d = [k for k in dense_features_1d.keys()]
    features_next.update({fc: tf.placeholder(dtype=tf.int64, shape=[None, 20], name=fc) for fc in \
                          sorted(list(dense_features_2d.keys()) + dense_features_1d_real_2d)})
    features_next.update({fc: tf.placeholder(dtype=tf.int64, shape=[None, 1], name=fc) for fc in \
                          sorted(dense_features_1d_real_1d)})
    return tf.estimator.export.ServingInputReceiver(features_next, features_next)


def run(_):
    model, _ = train_and_evaluate()
    if model.config.is_chief and FLAGS.is_train == 1:
        model.export_saved_model(
            export_dir_base=os.path.join(FLAGS.model_path, 'exported'),
            serving_input_receiver_fn=serving_input_receiver_fn
        )

if __name__ == '__main__':
    tf.app.run(run)