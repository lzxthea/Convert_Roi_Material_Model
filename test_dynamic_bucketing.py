# encoding: utf-8
"""
动态分桶功能测试脚本
"""

import tensorflow as tf
import numpy as np
import os
import json
from dynamic_bucketing_simple import SimpleDynamicBucketing, create_dynamic_bucket_feats_simple

def test_dynamic_bucketing():
    """
    测试动态分桶功能
    """
    print("=== 动态分桶功能测试 ===")
    
    # 1. 初始化动态分桶器
    print("1. 初始化动态分桶器...")
    dynamic_bucketing = SimpleDynamicBucketing(
        num_batches_for_stats=100,  # 使用100个batch进行测试
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        min_bucket_size=5,
        max_bucket_size=20,
        save_path="./test_bucket_points.json"
    )
    
    # 2. 模拟特征数据收集
    print("2. 收集特征统计信息...")
    feature_names = ['fc_send_cnt_30m_qc_product_id_qp_roi2', 
                     'fc_show_cnt_30m_qc_product_id_qp_roi2',
                     'fc_click_cnt_30m_qc_product_id_qp_roi2']
    
    batch_size = 100
    
    for batch_idx in range(100):  # 模拟100个batch
        # 生成模拟特征数据
        features = {}
        for feat_name in feature_names:
            if 'cnt' in feat_name:
                # 计数特征：大部分为0，少数为较大值
                values = np.random.exponential(scale=5.0, size=batch_size)
                values = np.where(np.random.random(batch_size) < 0.8, 0, values)
            else:
                # 其他特征：正态分布
                values = np.random.normal(10, 5, size=batch_size)
                values = np.maximum(values, 0)
            
            features[feat_name] = tf.constant(values.reshape(-1, 1), dtype=tf.float32)
        
        # 收集特征统计信息
        dynamic_bucketing.collect_feature_stats(features, feature_names)
        
        if batch_idx % 20 == 0:
            print(f"  已处理 {batch_idx} 个batch")
    
    print("3. 特征统计信息收集完成")
    
    # 3. 检查生成的分桶点
    print("4. 检查生成的分桶点...")
    all_bucket_points = dynamic_bucketing.get_all_bucket_points()
    
    for feat_name, bucket_points in all_bucket_points.items():
        print(f"  {feat_name}: {len(bucket_points)} 个分桶点")
        print(f"    分桶点: {bucket_points[:3]}...{bucket_points[-3:]}")
    
    # 4. 测试分桶点保存和加载
    print("5. 测试分桶点保存和加载...")
    
    # 创建新的分桶器实例
    new_bucketing = SimpleDynamicBucketing(
        num_batches_for_stats=100,
        save_path="./test_bucket_points.json"
    )
    
    # 加载分桶点
    loaded_points = new_bucketing.load_bucket_points()
    print(f"  加载了 {len(loaded_points)} 个特征的分桶点")
    
    # 5. 测试动态分桶处理
    print("6. 测试动态分桶处理...")
    
    # 原始静态分桶配置
    dense_features_1d = {
        'fc_send_cnt_30m_qc_product_id_qp_roi2': [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.0],
        'fc_show_cnt_30m_qc_product_id_qp_roi2': [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.1],
        'fc_click_cnt_30m_qc_product_id_qp_roi2': [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.2],
    }
    
    # 生成测试数据
    test_features = {}
    for feat_name in feature_names:
        if 'cnt' in feat_name:
            values = np.random.exponential(scale=5.0, size=batch_size)
            values = np.where(np.random.random(batch_size) < 0.8, 0, values)
        else:
            values = np.random.normal(10, 5, size=batch_size)
            values = np.maximum(values, 0)
        
        test_features[feat_name] = tf.constant(values.reshape(-1, 1), dtype=tf.float32)
    
    # 使用动态分桶处理
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        try:
            embeddings, names = create_dynamic_bucket_feats_simple(
                test_features, dense_features_1d, new_bucketing,
                log_dict={}, need_log1p=False, dim=4, all_feat_suffix="test"
            )
            
            print(f"  成功生成 {len(embeddings)} 个embedding")
            for i, (emb, name) in enumerate(zip(embeddings, names)):
                emb_value = sess.run(emb)
                print(f"    {name}: embedding shape {emb_value.shape}, 范围 [{emb_value.min():.4f}, {emb_value.max():.4f}]")
            
            print("✅ 动态分桶功能测试通过！")
            
        except Exception as e:
            print(f"❌ 动态分桶功能测试失败: {e}")
            return False
    
    # 6. 清理测试文件
    if os.path.exists("./test_bucket_points.json"):
        os.remove("./test_bucket_points.json")
        print("7. 清理测试文件完成")
    
    return True

def test_comparison():
    """
    对比静态分桶和动态分桶的效果
    """
    print("\n=== 静态分桶 vs 动态分桶对比 ===")
    
    # 生成测试数据
    batch_size = 1000
    feature_name = 'fc_send_cnt_30m_qc_product_id_qp_roi2'
    
    # 生成符合实际分布的数据
    values = np.random.exponential(scale=5.0, size=batch_size)
    values = np.where(np.random.random(batch_size) < 0.8, 0, values)
    
    features = {feature_name: tf.constant(values.reshape(-1, 1), dtype=tf.float32)}
    
    # 静态分桶点
    static_buckets = [0.001, 7.0, 11.0, 19.0, 26.0, 34.0, 42.0, 51.0, 60.0, 74.0]
    
    # 动态分桶点（模拟）
    dynamic_buckets = np.percentile(values[values > 0], [10, 20, 30, 40, 50, 60, 70, 80, 90])
    dynamic_buckets = dynamic_buckets[dynamic_buckets > 0].tolist()
    
    print(f"特征: {feature_name}")
    print(f"数据分布: 最小值={values.min():.3f}, 最大值={values.max():.3f}, 均值={values.mean():.3f}")
    print(f"静态分桶点: {static_buckets}")
    print(f"动态分桶点: {dynamic_buckets}")
    
    # 计算分桶效果
    def calculate_bucket_distribution(values, buckets):
        bucket_counts = np.zeros(len(buckets) + 1)
        for val in values:
            bucket_idx = 0
            for i, bucket in enumerate(buckets):
                if val <= bucket:
                    bucket_idx = i
                    break
            else:
                bucket_idx = len(buckets)
            bucket_counts[bucket_idx] += 1
        return bucket_counts
    
    static_dist = calculate_bucket_distribution(values, static_buckets)
    dynamic_dist = calculate_bucket_distribution(values, dynamic_buckets)
    
    print(f"\n静态分桶分布: {static_dist}")
    print(f"动态分桶分布: {dynamic_dist}")
    
    # 计算空桶数量
    static_empty = np.sum(static_dist == 0)
    dynamic_empty = np.sum(dynamic_dist == 0)
    
    print(f"\n静态分桶空桶数量: {static_empty}")
    print(f"动态分桶空桶数量: {dynamic_empty}")
    
    if dynamic_empty < static_empty:
        print("✅ 动态分桶减少了空桶数量，提高了分桶效率")
    else:
        print("⚠️ 动态分桶空桶数量未减少")

if __name__ == "__main__":
    # 运行测试
    success = test_dynamic_bucketing()
    
    if success:
        test_comparison()
        print("\n🎉 所有测试完成！")
    else:
        print("\n❌ 测试失败，请检查代码")
