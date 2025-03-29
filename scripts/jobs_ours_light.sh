mkdir -p ./logs/fig15/ogbn-arxiv
mkdir -p ./logs/fig16/ogbn-arxiv
mkdir -p ./logs/fig17/ogbn-arxiv
mkdir -p ./logs/fig18/Cora
mkdir -p ./logs/fig18/Reddit
mkdir -p ./logs/fig19/Cora
mkdir -p ./logs/fig19/Reddit

### Fig 15
# Ours (CMQ)
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.01.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.05 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.05.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.1 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.1.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.3 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.3.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.5 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.5.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.7 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.7.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.9 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.9.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.95 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.95.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.99 --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log --dim_hidden 192 --train_mode normal > ./logs/fig15/ogbn-arxiv/Ours_0.99.log 2>&1

### Fig 16
# Ours (CMQ)
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 2 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_2.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 4 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_4.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_8.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 16 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_16.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 32 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_32.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 64 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_64.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 128 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_128.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 256 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --fixed_centroid_ratio --centroid_ratio 0.01 --train_mode normal --dim_hidden 192 --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig16/ogbn-arxiv/Ours_256.log 2>&1

### Fig 17
# CMQ (online + hierarchical)
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 1 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_1.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 2 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_2.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 4 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_4.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 8 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_8.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 16 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_16.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 32 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_32.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 64 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_64.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 128 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_128.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 256 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_256.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 512 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_512.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 1024 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_1024.log 2>&1
python -m BingoGCN.graph_partitioning.main --dataset ogbn-arxiv --model GCN --n_parts 8 --validate --inter_cluster --outgoing_kmeans --hierarchical_kmeans --kmeans_distance l1 --dim_hidden 192 --num_kmeans_clusters 2048 --train_mode normal --init_mode kaiming_uniform --pretrained_log BingoGCN/pretrained_logs/SLT_structured_Dense_baseline/GCN/ogbn-arxiv/SLT_structured_Dense_baseline_GCN_HD192_ogbn-arxiv_L4_S0.log > ./logs/fig17/ogbn-arxiv/CMQ_online_hierarchical_2048.log 2>&1

### Fig 18
# FG Sparsity with SLT
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 64 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD64_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_64.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 96 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD96_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_96.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 128 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD128_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_128.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD192_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_192.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 256 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD256_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_256.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 384 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD384_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_384.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 512 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD512_Cora_Msup_L3_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig18/Cora/FG_sparsity_with_slt_512.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 64 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD64_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_64.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 96 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD96_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_96.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 128 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD128_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_128.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD192_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_192.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 256 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD256_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_256.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 384 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD384_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_384.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 512 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp1_HD_GCN_HD512_Reddit_Msup_L4_M16_S5000_FG --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --linear_sparsity 0.5625 --sparsity_list 0.562500 0.708333 0.854167 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig18/Reddit/FG_sparsity_with_slt_512.log 2>&1

### Fig19
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S625_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.062500 0.375000 0.687500 --linear_sparsity 0.0625 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S625.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S1250_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.125000 0.416667 0.708333 --linear_sparsity 0.125 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S1250.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S1875_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.187500 0.458333 0.729167 --linear_sparsity 0.1875 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S1875.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S2500_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.250000 0.500000 0.750000 --linear_sparsity 0.25 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S2500.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S3125_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.312500 0.541667 0.770833 --linear_sparsity 0.3125 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S3125.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S3750_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.375000 0.583333 0.791667 --linear_sparsity 0.375 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S3750.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S4375_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.437500 0.625000 0.812500 --linear_sparsity 0.4375 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S4375.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S5000_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.500000 0.666667 0.833333 --linear_sparsity 0.5 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S5000.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S5625_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.562500 0.708333 0.854167 --linear_sparsity 0.5625 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S5625.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S6250_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.625000 0.750000 0.875000 --linear_sparsity 0.625 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S6250.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S6875_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.687500 0.791667 0.895833 --linear_sparsity 0.6875 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S6875.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S7500_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.750000 0.833333 0.916667 --linear_sparsity 0.75 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S7500.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S8125_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.812500 0.875000 0.937500 --linear_sparsity 0.8125 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S8125.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S8750_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.875000 0.916667 0.958333 --linear_sparsity 0.875 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S8750.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 3 --dim_hidden 192 --dataset Cora --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Cora_Msup_L3_M16_S9375_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.937500 0.958333 0.979167 --linear_sparsity 0.9375 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm None > ./logs/fig19/Cora/S9375.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S625_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.062500 0.375000 0.687500 --linear_sparsity 0.0625 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S625.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S1250_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.125000 0.416667 0.708333 --linear_sparsity 0.125 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S1250.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S1875_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.187500 0.458333 0.729167 --linear_sparsity 0.1875 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S1875.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S2500_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.250000 0.500000 0.750000 --linear_sparsity 0.25 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S2500.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S3125_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.312500 0.541667 0.770833 --linear_sparsity 0.3125 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S3125.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S3750_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.375000 0.583333 0.791667 --linear_sparsity 0.375 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S3750.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S4375_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.437500 0.625000 0.812500 --linear_sparsity 0.4375 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S4375.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S5000_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.500000 0.666667 0.833333 --linear_sparsity 0.5 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S5000.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S5625_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.562500 0.708333 0.854167 --linear_sparsity 0.5625 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S5625.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S6250_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.625000 0.750000 0.875000 --linear_sparsity 0.625 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S6250.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S6875_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.687500 0.791667 0.895833 --linear_sparsity 0.6875 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S6875.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S7500_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.750000 0.833333 0.916667 --linear_sparsity 0.75 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S7500.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S8125_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.812500 0.875000 0.937500 --linear_sparsity 0.8125 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S8125.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S8750_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.875000 0.916667 0.958333 --linear_sparsity 0.875 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S8750.log 2>&1
python ./BingoGCN/main.py --command train --num_layers 4 --dim_hidden 192 --dataset Reddit --train_mode score_only --exp_name SLT_structured_exp2_sparsity_GCN_HD192_Reddit_Msup_L4_M16_S9375_--unstructured_for_last_layer_initsigned_constant_SF --epochs 400 --type_model GCN --repeat_times 3 --sparse_decay --init_mode signed_constant_SF --sparsity_list 0.937500 0.958333 0.979167 --linear_sparsity 0.9375 --unstructured_for_last_layer --enable_mask --nmsparsity --M 16 --type_norm batch > ./logs/fig19/Reddit/S9375.log 2>&1
