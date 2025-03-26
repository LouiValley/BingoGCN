from ..models.networks.GAT import GAT
from ..models.networks.GCN import GCN
from ..models.networks.gin_net import GINNet


def get_model(args, model_cfg=None, g=None):
    if model_cfg is None:
        model_cfg = "dm_GCN"  # 这里可以根据您的需求更改默认值
    if (
        model_cfg == "abs_GCN"
        or model_cfg == "abs_BN_GCN"
        or model_cfg == "high_sparse_abs_GCN"
        or model_cfg == "low_sparse_abs_GCN"
        or model_cfg == "high_sparse_BN_abs_GCN"
    ):
        model = GCN(args)
    elif (
        model_cfg == "abs_GAT"
        or model_cfg == "abs_BN_GAT"
        or model_cfg == "high_sparse_abs_GAT"
        or model_cfg == "low_sparse_abs_GAT"
        or model_cfg == "high_sparse_BN_abs_GAT"
        or model_cfg == "GAT"
    ):
        model = GAT(args)
    elif (
        model_cfg == "abs_GIN"
        or model_cfg == "abs_BN_GIN"
        or model_cfg == "high_sparse_abs_GIN"
        or model_cfg == "low_sparse_abs_GIN"
        or model_cfg == "high_sparse_BN_abs_GIN"
        or model_cfg == "GIN"
    ):
        model = GINNet(args, g)  # 注意：您需要定义变量g
    elif model_cfg == "GCN":
        # print('this is dmGCN for function model')
        model = GCN(args)
    return model
