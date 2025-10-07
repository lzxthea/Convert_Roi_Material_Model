python3 main.py \
--train_paths='hdfs://haruna/home/byte_ad_platform/user/liuzixi.123/ecp_roi2_nobid/pid_mid_anchor_model/multi_label_v1/20250918/part-0000*' \
--test_paths='hdfs://haruna/home/byte_ad_platform/user/liuzixi.123/ecp_roi2_nobid/pid_mid_anchor_model/multi_label_v1/20250919/part-0000*' \
--is_eval=0 \
--is_offline=1 \
--loss_type='weighted_cross_entropy,focal_loss' \
--save_summary_steps=1