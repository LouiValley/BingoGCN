import os
import re
import pandas as pd

root_directory = "./logs"

exp_directories_1 = [
    "fig19",
]
data_types_1 = [
    "ogbg-molbace",
    "ogbg-molhiv",
]
folder1 = "GCN_graph_UGTs"

for exp_dir in exp_directories_1:
    for data_type in data_types_1:
        directory = os.path.join(root_directory, exp_dir, data_type)

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
    "fig18",
    "fig19",
]
data_types_2 = [
    "Cora",
    "ogbn-arxiv",
    "Reddit",
    "Pubmed",
    "Citeseer",
]

for exp_dir in exp_directories_2:
    for data_type in data_types_2:
        directory = os.path.join(root_directory, exp_dir, data_type)

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
    "fig15",
    "fig16",
    "fig17",
]
data_types_3 = [
    "Cora",
    "ogbn-arxiv",
    "Reddit",
    "Pubmed",
    "Citeseer",
]

for exp_dir in exp_directories_3:
    for data_type in data_types_3:
        directory = os.path.join(root_directory, exp_dir, data_type)

        if not os.path.exists(directory):
            print(f"No directory: {directory}")
            continue

        data = []

        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                with open(os.path.join(directory, filename), "r") as file:
                    content = file.read()
                    test_acc_match = re.search(r" 'Test Acc': '(\d+\.\d+)'", content)
                    if test_acc_match:
                        data.append({
                            "Filename": filename,
                            "Test Acc": test_acc_match.group(1) if test_acc_match else None,
                        })

        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                with open(os.path.join(directory, filename), "r") as file:
                    content = file.read()
                    match = re.search(r"acc mean: (\d+\.\d+)", content)
                    if match:
                        data.append({
                            "Filename": filename,
                            "Test Acc": match.group(1),
                        })

        df = pd.DataFrame(data)
        if not df.empty:
            csv_path = os.path.join(directory, f"{exp_dir}_{data_type}_log_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV file: {csv_path}")
        else:
            print(f"No data: {directory}")
