import argparse
import datetime
import random
from pprint import PrettyPrinter

import commands
import numpy as np
import torch
import yaml
from base_options import BaseOptions

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hyperparam_names = {
    "lr": float,
    "weight_decay": float,
    "model.config_name": str,
    "seed": int,
    "conv_sparsity": float,
    "rerand_freq": int,
    "rerand_lambda": float,
}


def load_configs(config):
    with open(config, "r") as f:
        yml = f.read()
        dic = yaml.load(yml, Loader=yaml.FullLoader)
    return dic


def extract_values_from_dict_list(dict_list):
    values_list = []
    for d in dict_list:
        values_list.extend(d.values())
    return values_list


def main(args):
    pp = PrettyPrinter(indent=1)
    command = args.command
    print(args)
    # print(cfg)
    command = getattr(getattr(commands, command), command)
    results_acc = []
    results_ece = []
    for i in range(args.repeat_times):
        acc, ece = command(args.exp_name, args=args, idx=i)
        results_acc.append(acc)
        results_ece.append(ece)
    if isinstance(results_acc, list) and all(
        isinstance(d, dict) for d in results_acc
    ):
        extracted_values = extract_values_from_dict_list(results_acc)
        print(
            "acc mean:",
            np.mean(extracted_values),
            " acc std:",
            np.std(extracted_values),
        )
    else:
        print(
            "acc mean:",
            np.mean(results_acc),
            " acc std:",
            np.std(results_acc),
            "ece mean:",
            np.mean(results_ece),
            "ece std",
            np.std(results_ece),
        )


if __name__ == "__main__":
    # start time
    start_time = datetime.datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser(description="Constrained learing")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="number of training the one shot model",
    )
    parser.add_argument("--dim_hidden", type=int, default=192)
    parser.add_argument("--repeat_times", type=int, default=5)
    parser.add_argument("--linear_sparsity", type=float, default=0.1)
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        choices=["score_only", "normal"],
    )
    parser.add_argument("--rerand_freq", type=int, default=0)
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="specify the name of experiment",
    )
    parser.add_argument("--weight_l1", type=float, default=5e-4)
    parser.add_argument("--sampling", type=float, default=None)
    parser.add_argument(
        "--samplingtype",
        type=str,
        default=None,
        choices=[
            "RandomNodeSampler",
            "DegreeBasedSampler",
            "RandomEdgeSampler",
        ],
    )
    parser.add_argument("--sparse_decay", action="store_true")
    parser.add_argument(
        "--attack", type=str, default=None, choices=["features", "edges"]
    )
    parser.add_argument("--auroc", action="store_true")
    parser.add_argument("--attack_eps", type=float, default=0)
    args = BaseOptions().initialize(parser)
    main(args)

    # end time and dutation time
    end_time = datetime.datetime.now()
    print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    time_difference = end_time - start_time
    print("Duration:", time_difference)
