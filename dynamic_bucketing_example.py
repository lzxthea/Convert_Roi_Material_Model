# encoding: utf-8
"""
动态分桶使用示例
"""

import tensorflow as tf
import numpy as np
from dynamic_bucketing_simple import SimpleDynamicBucketing, create_dynamic_bucket_feats_simple

def example_usage():
    """
    动态分桶使用示例
    """
    # 1. 初始化动态分桶器
    dynamic_bucketing = SimpleDynamicBucketing(
        num_batches_for_stats=1000,  # 使用1000个batch统计分位点
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 分位点
        min_bucket_size=10,  # 最小桶数量
        max_bucket_size=50,  # 最大桶数量
        save_path="./bucket_points.json"  # 分桶点保存路径
    )
    
    # 2. 模拟特征数据
    batch_size = 512
    feature_names = ['fc_send_cnt_30m_qc_product_id_qp_roi2', 
                     'fc_show_cnt_30m_qc_product_id_qp_roi2',
                     'fc_click_cnt_30m_qc_product_id_qp_roi2']
    
    # 3. 模拟训练过程
    for batch_idx in range(1000):  # 模拟1000个batch
        # 生成模拟特征数据
        features = {}
        for feat_name in feature_names:
            # 生成符合实际分布的数据
            if 'cnt' in feat_name:
                # 计数特征：大部分为0，少数为较大值
                values = np.random.exponential(scale=5.0, size=batch_size)
                values = np.where(np.random.random(batch_size) < 0.8, 0, values)
            elif 'pvr' in feat_name or 'ectr' in feat_name:
                # 比率特征：0-1之间的值
                values = np.random.beta(2, 5, size=batch_size)
            else:
                # 其他特征：正态分布
                values = np.random.normal(10, 5, size=batch_size)
                values = np.maximum(values, 0)  # 确保非负
            
            features[feat_name] = tf.constant(values.reshape(-1, 1), dtype=tf.float32)
        
        # 收集特征统计信息
        dynamic_bucketing.collect_feature_stats(features, feature_names)
        
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx} batches")
    
    # 4. 使用动态分桶处理特征
    print("\n=== 使用动态分桶处理特征 ===")
    
    # 模拟新的batch数据
    test_features = {}
    for feat_name in feature_names:
        if 'cnt' in feat_name:
            values = np.random.exponential(scale=5.0, size=batch_size)
            values = np.where(np.random.random(batch_size) < 0.8, 0, values)
        else:
            values = np.random.normal(10, 5, size=batch_size)
            values = np.maximum(values, 0)
        
        test_features[feat_name] = tf.constant(values.reshape(-1, 1), dtype=tf.float32)
    
    # 原始静态分桶配置
    dense_features_1d = {
        'fc_send_cnt_30m_qc_product_id_qp_roi2': [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.0],
        'fc_show_cnt_30m_qc_product_id_qp_roi2': [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.1],
        'fc_click_cnt_30m_qc_product_id_qp_roi2': [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.2],
    }
    
    # 使用动态分桶处理
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        embeddings, names = create_dynamic_bucket_feats_simple(
            test_features, dense_features_1d, dynamic_bucketing,
            log_dict={}, need_log1p=False, dim=4, all_feat_suffix="test"
        )
        
        print(f"Generated {len(embeddings)} embeddings")
        for i, (emb, name) in enumerate(zip(embeddings, names)):
            emb_value = sess.run(emb)
            print(f"Feature {name}: embedding shape {emb_value.shape}")
    
    # 5. 查看生成的分桶点
    print("\n=== 生成的分桶点 ===")
    all_bucket_points = dynamic_bucketing.get_all_bucket_points()
    for feat_name, bucket_points in all_bucket_points.items():
        print(f"{feat_name}: {len(bucket_points)} buckets")
        print(f"  Points: {bucket_points[:5]}...{bucket_points[-5:]}")

if __name__ == "__main__":
    example_usage()
