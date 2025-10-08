# encoding: utf-8
"""
简化版动态分桶模块 - 基于1000个batch的特征分位点动态生成分桶点
"""

import tensorflow as tf
import numpy as np
from collections import defaultdict
import os
import json
from typing import Dict, List, Tuple, Optional

class SimpleDynamicBucketing:
    """
    简化版动态分桶类，基于训练数据的分位点动态生成分桶点
    """
    
    def __init__(self, 
                 num_batches_for_stats: int = 1000,
                 quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 min_bucket_size: int = 10,
                 max_bucket_size: int = 50,
                 save_path: str = "./bucket_points.json"):
        """
        初始化动态分桶器
        """
        self.num_batches_for_stats = num_batches_for_stats
        self.quantiles = quantiles
        self.min_bucket_size = min_bucket_size
        self.max_bucket_size = max_bucket_size
        self.save_path = save_path
        
        # 存储特征统计信息
        self.feature_stats = defaultdict(list)
        self.batch_count = 0
        self.bucket_points = {}
        self.is_initialized = False
        
    def collect_feature_stats(self, features: Dict[str, tf.Tensor], 
                            feature_names: List[str]) -> None:
        """
        收集特征统计信息
        """
        if self.is_initialized:
            return
            
        self.batch_count += 1
        
        for feat_name in feature_names:
            if feat_name in features:
                # 获取特征值
                feat_val = features[feat_name]
                if feat_val.shape[0] > 0:  # 确保batch不为空
                    # 处理1D特征
                    if len(feat_val.shape) == 2 and feat_val.shape[1] == 1:
                        feat_values = tf.reshape(feat_val, [-1])
                    else:
                        feat_values = feat_val
                    
                    # 过滤异常值
                    feat_values = tf.where(
                        tf.logical_and(
                            tf.greater_equal(feat_values, 0),
                            tf.less_equal(feat_values, 1e6)  # 过滤极大值
                        ),
                        feat_values,
                        tf.zeros_like(feat_values)
                    )
                    
                    # 收集非零值
                    non_zero_values = tf.boolean_mask(feat_values, tf.greater(feat_values, 0))
                    if tf.size(non_zero_values) > 0:
                        self.feature_stats[feat_name].append(non_zero_values)
        
        # 当收集到足够batch时，计算分桶点
        if self.batch_count >= self.num_batches_for_stats:
            self._compute_bucket_points()
            self.is_initialized = True
            self._save_bucket_points()
    
    def _compute_bucket_points(self) -> None:
        """
        基于收集的统计信息计算分桶点
        """
        for feat_name, feat_values_list in self.feature_stats.items():
            if not feat_values_list:
                continue
                
            # 合并所有batch的特征值
            all_values = tf.concat(feat_values_list, axis=0)
            
            # 计算分位点
            values_np = tf.keras.backend.get_value(all_values)
            quantile_values = []
            for q in self.quantiles:
                quantile_val = np.percentile(values_np, q * 100)
                quantile_values.append(quantile_val)
            
            # 排序并过滤
            quantile_values = sorted([v for v in quantile_values if v > 0])
            
            # 去重并限制桶数量
            unique_quantiles = []
            prev_val = -1
            for val in quantile_values:
                if val - prev_val > 1e-6:  # 避免重复值
                    unique_quantiles.append(val)
                    prev_val = val
            
            # 确保桶数量在合理范围内
            if len(unique_quantiles) < self.min_bucket_size:
                # 如果分位点太少，使用等间距分桶
                min_val = np.min(values_np)
                max_val = np.max(values_np)
                unique_quantiles = np.linspace(min_val, max_val, self.min_bucket_size).tolist()
            elif len(unique_quantiles) > self.max_bucket_size:
                # 如果分位点太多，选择关键分位点
                indices = np.linspace(0, len(unique_quantiles)-1, self.max_bucket_size, dtype=int)
                unique_quantiles = [unique_quantiles[i] for i in indices]
            
            self.bucket_points[feat_name] = unique_quantiles
            
            print(f"Generated {len(unique_quantiles)} bucket points for {feat_name}")
            print(f"Bucket points: {unique_quantiles[:5]}...{unique_quantiles[-5:]}")
    
    def _save_bucket_points(self) -> None:
        """
        保存分桶点到文件
        """
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.bucket_points, f, indent=2)
        print(f"Bucket points saved to {self.save_path}")
    
    def load_bucket_points(self) -> Dict[str, List[float]]:
        """
        从文件加载分桶点
        """
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.bucket_points = json.load(f)
            self.is_initialized = True
            print(f"Loaded bucket points from {self.save_path}")
        return self.bucket_points
    
    def get_bucket_points(self, feat_name: str) -> List[float]:
        """
        获取指定特征的分桶点
        """
        if not self.is_initialized:
            raise ValueError("Dynamic bucketing not initialized. Call collect_feature_stats first.")
        
        return self.bucket_points.get(feat_name, [])
    
    def get_all_bucket_points(self) -> Dict[str, List[float]]:
        """
        获取所有特征的分桶点
        """
        return self.bucket_points.copy()


def create_dynamic_bucket_feats_simple(features: Dict[str, tf.Tensor], 
                                      dense_features_1d: Dict[str, List[float]],
                                      dynamic_bucketing: SimpleDynamicBucketing,
                                      log_dict: Optional[Dict] = None,
                                      dim: int = 1,
                                      need_log1p: bool = True,
                                      all_feat_suffix: str = "common") -> Tuple[List[tf.Tensor], List[str]]:
    """
    使用动态分桶处理1D特征 - 简化版
    """
    if log_dict is None:
        log_dict = {}
    
    state_embeddings = []
    state_embeddings_names = []
    
    for feat_name in sorted(dense_features_1d.keys()):
        if feat_name not in features:
            print(f"Warning: Feature {feat_name} not found in features")
            continue
        
        # 获取动态分桶点
        dynamic_bucket_points = dynamic_bucketing.get_bucket_points(feat_name)
        
        if not dynamic_bucket_points:
            print(f"Warning: No dynamic bucket points for {feat_name}, using original")
            bucket_points = dense_features_1d[feat_name]
        else:
            bucket_points = dynamic_bucket_points
        
        # 处理特征值
        feat_val = features[feat_name]
        if len(feat_val.shape) == 2 and feat_val.shape[1] == 1:
            feat_val = feat_val[:, 0]
        
        # 应用log变换
        if need_log1p and feat_name not in log_dict:
            feat_val = tf.maximum(tf.cast(feat_val, tf.float32), 0.0)
            feat_val = tf.log1p(feat_val)
        else:
            feat_val = tf.cast(feat_val, tf.float32)
        
        if feat_name in log_dict:
            feat_val = tf.maximum(tf.cast(feat_val, tf.float32), 0.0)
            feat_val = tf.log1p(feat_val)
            bucket_points = log_dict[feat_name]
        
        # 使用动态分桶点进行分桶
        bucket_num = len(bucket_points) + 1
        bucket_info = tf.raw_ops.Bucketize(
            input=feat_val, boundaries=bucket_points
        )
        
        # 创建embedding权重
        weights = tf.get_variable(
            name=f"{feat_name}_weights",
            shape=[bucket_num, dim],
            initializer=tf.random_uniform_initializer(-0.0018125, 0.0018125)
        )
        
        # 获取embedding
        emb_tensor = tf.gather(weights, bucket_info, axis=0)
        emb_tensor = tf.reshape(emb_tensor, shape=[-1, dim])
        
        state_embeddings.append(emb_tensor)
        state_embeddings_names.append(feat_name)
        
        # 添加统计信息
        tf.summary.scalar(f'raw_feat_mean_{feat_name}_{all_feat_suffix}', 
                         tf.sqrt(tf.reduce_mean(feat_val)))
        tf.summary.histogram(f'bucket_info_{feat_name}_{all_feat_suffix}', bucket_info)
    
    print(f"Dynamic bucketing: {len(state_embeddings)} features processed")
    return state_embeddings, state_embeddings_names
