import math

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

# import torch.nn as nn

# cls_criterion = torch.nn.BCEWithLogitsLoss()
# # reg_criterion = torch.nn.MSELoss()


def get_sparsity(sparsity, current_epoches, start_epoches, end_epoches):
    sparsity = sparsity - sparsity * (
        1
        - (current_epoches + 1 - start_epoches)
        * 1.0
        / (end_epoches - start_epoches)
    )
    return sparsity


class GraphLevelLearning(object):
    def __init__(self, outman, args, device, model):
        self.args = args
        # print(self.args)
        self.outman = outman
        # self.cfg = cfg
        self.device = device
        self.data_parallel = False

        self.model = model
        self.optimizer = self._get_optimizer(args.train_mode, self.model)
        # self.lr_scheduler = self._get_scheduler()
        self.criterion = self._get_criterion()
        self.scheduler = self._get_scheduler()

    def train(self, epoch, loader, model, evaluator):
        model.train()
        # total_count = 0
        # total_loss = 0.0
        # correct = 0
        all_preds = []
        all_labels = []
        # step_before_train = (
        #     hasattr(self.scheduler, "step_before_train")
        #     and self.scheduler.step_before_train
        # )
        # if step_before_train:
        #     try:
        #         self.scheduler.step()
        #     except:
        #         self.scheduler.step()
        half_epochs = self.args.epochs / 2.0
        use_initial_sparsity = self.args.sparse_decay and epoch < half_epochs
        if (
            self.args.enable_mask and self.args.local_pruning is False
        ):  # global_mm
            sparsity = [
                (
                    get_sparsity(vs, epoch, 0, half_epochs)
                    if use_initial_sparsity
                    else vs
                )
                for vs in self.args.sparsity_list
            ]
        elif (
            self.args.enable_mask and self.args.local_pruning is True
        ):  # local_mm
            sparsity = [
                [
                    (
                        get_sparsity(vs_val, epoch, 0, half_epochs)
                        if use_initial_sparsity
                        else vs_val
                    )
                    for vs_val in vs_group
                ]
                for vs_group in self.args.local_sparsity_list
            ]
        elif self.args.local_pruning is True:  # local_pruning_sm
            sparsity = [
                (
                    get_sparsity(vs, epoch, 0, half_epochs)
                    if use_initial_sparsity
                    else vs
                )
                for vs in self.args.local_sparsity_list
            ]
        else:  # global_pruning_sm
            sparsity = (
                get_sparsity(self.args.linear_sparsity, epoch, 0, half_epochs)
                if use_initial_sparsity
                else self.args.linear_sparsity
            )

        for step, batch in enumerate(loader):
            batch = batch.to(self.device)

            # if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            #     continue
            # else:
            # self.model.train()

            pred = model(batch, sparsity, epoch=epoch)

            # Assume batch.y contains the true labels
            is_labeled = (
                batch.y == batch.y
            )  # Check which data points have labels

            if self.args.dataset == "hep10k":
                loss = self.criterion(
                    pred.to(torch.float32)[is_labeled],
                    pred.to(torch.float32)[is_labeled],
                )
            else:
                loss = self.criterion(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if not step_before_train:
            #     try:
            #         self.scheduler.step()
            #     except:
            #         self.scheduler.step()

            # Store predictions and labels for evaluation
            all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
            if self.args.dataset == "hep10k":
                all_labels.append(pred.cpu().detach().numpy())
            else:
                all_labels.append(batch.y.cpu().numpy())

            # Calculate correct predictions
            # _, predicted = pred.max(
            #     1
            # )  # Assuming pred is output logits and we need the index of the max logit
            # correct += (
            #     (predicted == batch.y).sum().item()
            # )  # Count correct predictions
            # total_count += (
            #     is_labeled.sum().item()
            # )  # Count all labeled examples
            # total_loss += loss.item()

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate evaluation metrics using the evaluator
        input_dict = {"y_true": all_labels, "y_pred": all_preds}

        return evaluator.eval(input_dict)

    def evaluate(self, loader, model, evaluator):
        model.eval()
        y_true = []
        y_pred = []
        all_preds = []
        all_labels = []
        for step, batch in enumerate(loader):
            batch = batch.to(self.device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    # pred = model(batch, sparsity, epoch=self.args.epochs)
                    pred = model(batch, sparsity=None, epoch=self.args.epochs)

                if self.args.dataset == "hep10k":
                    y_true.append(pred.detach().cpu())
                    y_pred.append(pred.detach().cpu())
                    all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
                    all_labels.append(torch.sigmoid(pred).cpu().detach().numpy())
                else:
                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                    y_pred.append(pred.detach().cpu())
                    all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
                    all_labels.append(batch.y.cpu().numpy())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # if evaluator == Test_evaluator:
        # roc_auc = roc_auc_score(all_labels, all_preds)
        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)

    def _get_optimizer(self, mode, model):
        # print("mode",mode)
        optim_name = self.args.optimizer
        # for name, param in model.named_parameters():
        #    print(name)
        if mode == "score_only":
            lr = self.args.lr
            weight_decay = self.args.weight_decay

            params = [
                param
                for param in model.parameters()
                if (hasattr(param, "is_score") and param.is_score)
            ]
            # print("update params")
            # for name, param in model.named_parameters():
            #     if hasattr(param, "is_score") and param.is_score:
            #         print(name)

            return self._new_optimizer(optim_name, params, lr, weight_decay)

        elif mode == "normal":
            lr = self.args.lr
            weight_decay = self.args.weight_decay
            params = [
                param
                for param in self.model.parameters()
                if not (hasattr(param, "is_score") and param.is_score)
            ]
            # print(params)
            return self._new_optimizer(optim_name, params, lr, weight_decay)
        else:
            raise NotImplementedError

    def _get_criterion(self):

        if self.args.type_model == "GCN_graph_UGTs":
            return torch.nn.BCEWithLogitsLoss()
        elif self.args.dataset == "Reddit":
            return torch.nn.NLLLoss()
        else:
            return torch.nn.CrossEntropyLoss()

    def _new_optimizer(self, name, params, lr, weight_decay, momentum=0.9):
        if name == "Adam":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == "SGD":
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=self.args.sgd_momentum,
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError

    def _get_scheduler(self):
        class null_scheduler(object):
            def __init__(self, *args, **kwargs):
                return

            def step(self, *args, **kwargs):
                return

            def state_dict(self):
                return {}

            def load_state_dict(self, dic):
                return

        class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(
                self,
                optimizer: torch.optim.Optimizer,
                warmup_epochs: int,
                max_epochs: int,
                warmup_start_lr: float = 0.00001,
                eta_min: float = 0.00001,
                last_epoch: int = -1,
            ):
                """
                Args:
                    optimizer (torch.optim.Optimizer):
                        最適化手法インスタンス
                    warmup_epochs (int):
                        linear warmupを行うepoch数
                    max_epochs (int):
                        cosine曲線の終了に用いる 学習のepoch数
                    warmup_start_lr (float):
                        linear warmup 0 epoch目の学習率
                    eta_min (float):
                        cosine曲線の下限
                    last_epoch (int):
                        cosine曲線の位相オフセット
                学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする
                epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる
                https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
                """
                self.warmup_epochs = warmup_epochs
                self.max_epochs = max_epochs
                self.warmup_start_lr = warmup_start_lr
                self.eta_min = eta_min
                super().__init__(optimizer, last_epoch)
                return None

            def get_lr(self):
                if self.last_epoch == 0:
                    return [self.warmup_start_lr] * len(self.base_lrs)
                if self.last_epoch < self.warmup_epochs:
                    return [
                        group["lr"]
                        + (base_lr - self.warmup_start_lr)
                        / (self.warmup_epochs - 1)
                        for base_lr, group in zip(
                            self.base_lrs, self.optimizer.param_groups
                        )
                    ]
                if self.last_epoch == self.warmup_epochs:
                    return self.base_lrs
                if (self.last_epoch - 1 - self.max_epochs) % (
                    2 * (self.max_epochs - self.warmup_epochs)
                ) == 0:
                    return [
                        group["lr"]
                        + (base_lr - self.eta_min)
                        * (
                            1
                            - math.cos(
                                math.pi
                                / (self.max_epochs - self.warmup_epochs)
                            )
                        )
                        / 2
                        for base_lr, group in zip(
                            self.base_lrs, self.optimizer.param_groups
                        )
                    ]

                return [
                    (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_epochs)
                            / (self.max_epochs - self.warmup_epochs)
                        )
                    )
                    / (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_epochs - 1)
                            / (self.max_epochs - self.warmup_epochs)
                        )
                    )
                    * (group["lr"] - self.eta_min)
                    + self.eta_min
                    for group in self.optimizer.param_groups
                ]

        if self.args.scheduler is None:
            return null_scheduler()
        elif self.args.scheduler == "CustomCosineLR":
            total_epoch = self.args.epochs
            init_lr = self.args.lr
            # warmup_epochs = self.args.warmup_epochs
            # ft_epochs = self.args.finetuning_epochs
            # ft_lr = self.args.finetuning_lr
            return CosineAnnealingLR(
                self.optimizer,
                max_epochs=total_epoch,
                warmup_epochs=total_epoch**0.03,
                warmup_start_lr=init_lr,
                eta_min=0.00001,
            )
        elif self.args.scheduler == "MultiStepLR":
            return MultiStepLR(
                self.optimizer,
                milestones=self.args.lr_milestones,
                gamma=self.args.multisteplr_gamma,
            )
        else:
            raise NotImplementedError
