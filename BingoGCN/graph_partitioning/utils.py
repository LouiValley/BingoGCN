import os
import re


def extract_log_data(logfile_path):
    # 检查文件是否存在
    if not os.path.exists(logfile_path):
        print(f"Log file not found: {logfile_path}")
        return None, None, None, None  # 或其他适当的默认值

    with open(logfile_path, "r") as f:
        lines = f.readlines()

    # 反转行列表以从文件末尾开始读取
    lines = lines[::-1]

    # 初始化变量
    acc_mean, acc_std, best_acc = None, None, None
    model_path = None

    # 正则表达式来匹配所需的行
    acc_pattern = re.compile(r"acc mean: (\d+\.\d+)\s+acc std: (\d+\.\d+)")
    best_acc_pattern = re.compile(r"best acc: (\d+\.\d+)")
    model_path_pattern = re.compile(r"(.+\.pth)")

    # 遍历文件的每一行，匹配需要的数据
    for line in lines:
        if "acc mean:" in line:
            match = acc_pattern.search(line)
            if match:
                acc_mean, acc_std = match.groups()
        if "best acc:" in line:
            match = best_acc_pattern.search(line)
            if match:
                best_acc = match.group(1)
        if ".pth" in line and model_path is None:
            match = model_path_pattern.search(line)
            if match:
                model_path = match.group(1).replace("dump", "best")
                break  # 假定模型路径在文件末尾，找到后即可停止遍历

    return acc_mean, acc_std, best_acc, model_path

def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values


def logread(logfile_path):
    with open(logfile_path, "r") as f:
        lines = f.readlines()
        last_line = lines[-1]
        model_path = last_line.split("saved as ")[1].strip()
    return model_path


def get_output_path(args, dataset, model_cfg):
    base_path = "./BingoGCN/logs/"
    sampling_param = ""

    if args.every_X_is_approximated:
        sampling_method = "every_X_is_approximated-"
        sampling_param += f"{args.num_kmeans_clusters}"
    elif args.random_sampling:
        sampling_method = "random"
        sampling_param += args.inter_sparsity
    elif args.topk_sampling:
        sampling_method = "topk"
        sampling_param += args.inter_sparsity
    elif args.partial_kmeans:
        sampling_method = "partial_kmeans"
        sampling_param += f"{args.num_kmeans_clusters}"
    elif args.global_kmeans:
        sampling_method = "global_kmeans"
        sampling_param += f"{args.num_kmeans_clusters}"
    elif args.outgoing_kmeans:
        sampling_method = "outgoing_kmeans"
        sampling_param += f"{args.num_kmeans_clusters}"
    elif args.no_inter_cluster:
        sampling_method = "no_inter"
    else:
        sampling_method = "all_inter"

    file_name = f"{dataset}_{model_cfg}_w{args.linear_sparsity}-p{args.n_parts}-inter-{sampling_method}"
    if sampling_param:
        file_name += f"{sampling_param}"
    if args.online_kmeans:
        file_name += "_online"
    if args.streamingMETIS:
        file_name += "_streamingMETIS"
    file_name += ".xlsx"

    return os.path.join(base_path, file_name)
