import os
import re
import pandas as pd

root_directory = "./logs"

exp_directories_1 = [
    "Graph-level-SLT_structured_exp2_sparsity",
]
data_types_1 = [
    "ogbg-molbace",
    "ogbg-molhiv",
]
folder1 = "GCN_graph_UGTs"

for exp_dir in exp_directories_1:
    for data_type in data_types_1:
        directory = os.path.join(root_directory, exp_dir, folder1, data_type)

        if not os.path.exists(directory):
            print(f"No directory: {directory}")
            continue

        data = []

        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                with open(os.path.join(directory, filename), "r") as file:
                    content = file.read()
                    match = re.search(r"acc mean: (\d+\.\d+)", content)
                    if match:
                        data.append({
                            "Filename": filename,
                            "Acc Mean": match.group(1),
                        })

        df = pd.DataFrame(data)
        if not df.empty:
            csv_path = os.path.join(directory, f"{exp_dir}_{data_type}_log_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV file: {csv_path}")
        else:
            print(f"No data: {directory}")

exp_directories_2 = [
    "SLT_structured_Dense_baseline",
    "SLT_structured_exp1_HD",
    "SLT_structured_exp1_HD_rebuttal",
    "SLT_structured_exp2_sparsity",
    "SLT_structured_exp3_M",
    "SLT_structured_exp4_initialization",
    "SLT_structured_exp5_last_layer",
    "SLT_structured_exp6_num_layer",
]
data_types_2 = [
    "Cora",
    "ogbn-arxiv",
    "Reddit",
    "Pubmed",
    "Citeseer",
]
folder2 = "GCN"

for exp_dir in exp_directories_2:
    for data_type in data_types_2:
        directory = os.path.join(root_directory, exp_dir, folder2, data_type)

        if not os.path.exists(directory):
            print(f"No directory: {directory}")
            continue

        data = []

        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                with open(os.path.join(directory, filename), "r") as file:
                    content = file.read()
                    match = re.search(r"acc mean: (\d+\.\d+)", content)
                    if match:
                        data.append({
                            "Filename": filename,
                            "Acc Mean": match.group(1),
                        })

        df = pd.DataFrame(data)
        if not df.empty:
            csv_path = os.path.join(directory, f"{exp_dir}_{data_type}_log_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV file: {csv_path}")
        else:
            print(f"No data: {directory}")

exp_directories_3 = [
    "Partition_exp1_num_partition_nodes",
    "Partition_exp3_fixed_centroid_ratio",
    "Partition_exp4_num_L2_centroid",
    "Partition_exp4_num_L2_centroid_l1dist",
    "Partition_exp5-1_random_topk",
    "Partition_exp5-2_offline_online_hkmeans",
    "Partition_exp8_num_L1_centroid_SLT",
    "Partition_exp1_num_partition_nodes_SLT",
    "Partition_exp3_fixed_centroid_ratio_SLT",
    "Partition_exp4_num_L2_centroid_SLT",
    "Partition_exp4_num_L2_centroid_l1dist_SLT",
    "Partition_exp5-1_random_topk_SLT",
    "Partition_exp5-2_offline_online_hkmeans_SLT",
    "Partition_exp7_n_parts_vs_acc_in_fixed_centriod_ratio",
    "Partition_exp7_n_parts_vs_acc_in_fixed_centriod_ratio_SLT",
    "Partition_exp8_num_L1_centroid_SLT",
]
data_types_3 = [
    "Cora",
    "ogbn-arxiv",
    "Reddit",
    "Pubmed",
    "Citeseer",
]
folder3 = "GCN"

for exp_dir in exp_directories_3:
    for data_type in data_types_3:
        directory = os.path.join(root_directory, exp_dir, folder3, data_type)

        if not os.path.exists(directory):
            print(f"No directory: {directory}")
            continue

        data = []

        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                with open(os.path.join(directory, filename), "r") as file:
                    content = file.read()
                    test_acc_match = re.search(r" 'Test Acc': '(\d+\.\d+)'", content)
                    no_partition_acc_match = re.search(r"'No partition Test Acc': '(\d+\.\d+)'", content)
                    if test_acc_match or no_partition_acc_match:
                        data.append({
                            "Filename": filename,
                            "Test Acc": test_acc_match.group(1) if test_acc_match else None,
                            "No Partition Test Acc": no_partition_acc_match.group(1) if no_partition_acc_match else None,
                        })

        df = pd.DataFrame(data)
        if not df.empty:
            csv_path = os.path.join(directory, f"{exp_dir}_{data_type}_log_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV file: {csv_path}")
        else:
            print(f"No data: {directory}")
