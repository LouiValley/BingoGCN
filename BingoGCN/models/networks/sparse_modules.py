import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from datetime import datetime

EPSILON = np.finfo(float).eps


def calculate_and_plot_sparsity(weight, name):
    # sparsityの計算（PyTorchの関数を使用）
    sparsity = torch.mean((weight == 0).float(), axis=0)
    filename = (
        f"sparsity_distribution_{name}_{weight.shape[0]}_{weight.shape[1]}.png"
    )
    # sparsityの分布をヒストグラムとして描画
    plt.figure(figsize=(10, 6))
    # CUDAテンソルをCPUに移動させ、NumPy配列に変換
    plt.hist(sparsity.cpu().numpy(), bins=100, color="blue", edgecolor="black")
    plt.title("Sparsity Distribution")
    plt.xlabel("Sparsity")
    plt.ylabel("Frequency")

    # 画像として保存
    plt.savefig(filename)
    plt.close()


# def calculate_and_plot_sparsity(weight, name):

#     # sparsityの計算（PyTorchの関数を使用）
#     sparsity = torch.mean((weight == 0).float(), axis=0)
#     filename = (
#         f"sparsity_distribution_{name}_{weight.shape[0]}_{weight.shape[1]}.png"
#     )
#     # sparsityの分布をヒストグラムとして描画
#     plt.figure(figsize=(10, 6))
#     # CUDAテンソルをCPUに移動させ、NumPy配列に変換
#     plt.hist(sparsity.cpu().numpy(), bins=100, color="blue", edgecolor="black")
#     plt.title("Sparsity Distribution")
#     plt.xlabel("Sparsity")
#     plt.ylabel("Frequency")

#     # 画像として保存
#     plt.savefig(filename)
#     plt.close()


def calculate_sparsity(x):
    # GPU上のテンソルをCPUに移動し、勾配情報を切り離した後、NumPy配列に変換
    x_cpu = x.detach().cpu().numpy()  # xがGPU上にある場合に必要
    # 0の数をカウント
    zero_count = np.count_nonzero(x_cpu == 0)
    # 全要素数
    total_elements = x_cpu.size
    # sparsityの計算
    sparsity_ratio = zero_count / total_elements
    return sparsity_ratio


# def sparse_mul_counter(adj, x):
#     if len(adj.shape) == 3:
#         adj = torch.squeeze(adj, 0)
#     if len(x.shape) == 3:
#         x = torch.squeeze(x, 0)
#     sum = torch.spmm(adj, x)
#     non_zero = torch.count_nonzero(sum)
#     adj_sparse = adj.to_sparse()
#     x_sparse = x.to_sparse()

#     n_mul_tot = 0  # 非零乘法次数

#     # 遍历adj的每一行
#     for i in range(adj_sparse.shape[0]):
#         row_indices = adj_sparse.indices()[0] == i  # 找到第i行的非零元素的索引
#         cols = adj_sparse.indices()[1][row_indices]  # 对应的列索引

#         # 对于每一列，计算与x中相应列的非零元素乘法次数
#         for col in cols:
#             n_mul_tot += x_sparse.indices()[1][x_sparse.indices()[0] == col].size(0)
#     MACs = n_mul_tot + n_mul_tot - non_zero
#     return MACs.item()


def flip_odd(array, even=False):
    rev = 1 if even else 0
    shape = array.shape
    tmp = []
    for i in range(shape[2]):
        if i % 2 == rev:
            tmp.append(array[:, :, i, :])
        else:
            tmp.append(array[:, :, i, ::-1])
    return np.concatenate(tmp, axis=2).reshape(shape)


def padding_ch(array, stride=1):
    minch = 16 if stride == 1 else 4

    add_oc = (minch - (array.shape[0] % minch)) % minch
    add_ic = (minch - (array.shape[1] % minch)) % minch
    array = np.pad(array, (((0, add_oc), (0, add_ic), (0, 0), (0, 0))))

    return array


def padding_st2(array):
    if array.shape[2] % 2 == 1:
        array = np.pad(array, (((0, 0), (0, 0), (1, 0), (0, 0))))
    if array.shape[3] % 2 == 1:
        array = np.pad(array, (((0, 0), (0, 0), (0, 0), (1, 0))))
    return array


def trans_st2(array):
    print(array.shape)
    shape = array.shape
    tmp_shape = [
        shape[0],
        shape[1],
        math.ceil(shape[2] / 2),
        2,
        math.ceil(shape[3] / 2),
        2,
    ]
    out_shape = [
        shape[0],
        shape[1] * 4,
        math.ceil(shape[2] / 2),
        math.ceil(shape[3] / 2),
    ]

    array = array.reshape(tmp_shape).transpose((0, 1, 3, 5, 2, 4))
    array = array.reshape(out_shape)
    return array


def trans_st2_reverse(array):
    shape = array.shape
    tmp_shape = [shape[0], int(shape[1] / 4), 2, 2, shape[2], shape[3]]
    out_shape = [shape[0], int(shape[1] / 4), shape[2] * 2, shape[3] * 2]

    array = array.reshape(tmp_shape).transpose((0, 1, 4, 2, 5, 3))
    array = array.reshape(out_shape)
    return array


def xorshift_atom(generator, init_seed=None):
    ret = init_seed

    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator(*ret)
        return ret[-1]

    return inner


def xor16_atom(y=1234):
    # print(y)
    y = y ^ (y << 7 & 0xFFFF)
    y = y ^ (y >> 9 & 0xFFFF)
    y = y ^ (y << 8 & 0xFFFF)
    return (y & 0xFFFF,)


def maketensor_xor16_ko_atom(shape, layer=1, stride=1, flip=False, args=None):
    assert stride < 3, "not implemented error!"

    orig_shape = shape

    tmp_shape = []
    for i in range(len(shape)):
        tmp_shape.append(shape[i])
    shape = tmp_shape
    bchannel = 16 if stride == 1 else 4
    shape[0] = math.ceil(shape[0] / bchannel) * bchannel
    if stride == 2:
        if shape[2] % 2 == 1:
            shape[2] += 1
            pad_h = 1
        else:
            pad_h = 0
        if shape[3] % 2 == 1:
            shape[3] += 1
            pad_w = 1
        else:
            pad_w = 0
        shape = [shape[0] * 4, shape[1], int(shape[2] / 2), int(shape[3] / 2)]

    tensor = np.zeros(shape, dtype=int)

    for i in range(shape[1]):
        seed = ((layer + 1) << 10) ^ i  # layer means list_layer
        random16 = xorshift_atom(xor16_atom, init_seed=(seed,))
        for j in range(int(shape[0] / 16)):
            for k in range(shape[2] * shape[3]):
                out = random16()
                for _l in range(16):
                    tensor[j * 16 + _l][i][k // shape[2]][k % shape[3]] = int(
                        (out & (2**_l)) != 0
                    )

    if stride == 1:
        if flip is True:
            tensor = flip_odd(tensor)
        return tensor[: orig_shape[0], :, :, :]

    elif stride == 2:
        tensor = flip_odd(tensor)
        tensor = trans_st2_reverse(tensor)
        return tensor[: orig_shape[0], :, pad_h:, pad_w:]


def xorshift_bond(generator, init_seed=None):
    ret = init_seed

    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator(*ret)
        return ret[-1]

    return inner


def xor16_bond(y=1234):
    # print(y)
    y = y ^ (y << 7 & 0xFFFF)
    y = y ^ (y >> 9 & 0xFFFF)
    y = y ^ (y << 8 & 0xFFFF)
    return (y & 0xFFFF,)


def maketensor_xor16_ko_bond(
    shape, layer=1, stride=1, flip=False, list_num=None, args=None
):
    assert stride < 3, "not implemented error!"

    orig_shape = shape

    tmp_shape = []
    for i in range(len(shape)):
        tmp_shape.append(shape[i])
    shape = tmp_shape
    bchannel = 16 if stride == 1 else 4
    shape[0] = math.ceil(shape[0] / bchannel) * bchannel
    if stride == 2:
        if shape[2] % 2 == 1:
            shape[2] += 1
            pad_h = 1
        else:
            pad_h = 0
        if shape[3] % 2 == 1:
            shape[3] += 1
            pad_w = 1
        else:
            pad_w = 0
        shape = [shape[0] * 4, shape[1], int(shape[2] / 2), int(shape[3] / 2)]

    tensor = np.zeros(shape, dtype=int)

    for i in range(shape[1]):
        seed = ((layer + list_num + 1) << 10) ^ i  # layer means list_layer
        random16 = xorshift_bond(xor16_bond, init_seed=(seed,))
        for j in range(int(shape[0] / 16)):
            for k in range(shape[2] * shape[3]):
                out = random16()
                for _l in range(16):
                    tensor[j * 16 + _l][i][k // shape[2]][k % shape[3]] = int(
                        (out & (2**_l)) != 0
                    )

    if stride == 1:
        if flip is True:
            tensor = flip_odd(tensor)
        return tensor[: orig_shape[0], :, :, :]

    elif stride == 2:
        tensor = flip_odd(tensor)
        tensor = trans_st2_reverse(tensor)
        return tensor[: orig_shape[0], :, pad_h:, pad_w:]


def xorshift_offset(generator, init_seed=None):
    ret = init_seed

    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator(*ret)
        return ret[-1]

    return inner


def xor16_offset(y=1234):
    y = y ^ (y << 7 & 0xFFFF)
    y = y ^ (y >> 9 & 0xFFFF)
    y = y ^ (y << 8 & 0xFFFF)
    return (y & 0xFFFF,)


def maketensor_xor16_ko_offset(
    shape, layer=1, stride=1, flip=False, args=None
):
    assert stride < 3, "not implemented error!"

    orig_shape = shape

    tmp_shape = []
    for i in range(len(shape)):
        tmp_shape.append(shape[i])
    shape = tmp_shape
    bchannel = 16 if stride == 1 else 4
    shape[0] = math.ceil(shape[0] / bchannel) * bchannel
    if stride == 2:
        if shape[2] % 2 == 1:
            shape[2] += 1
            pad_h = 1
        else:
            pad_h = 0
        if shape[3] % 2 == 1:
            shape[3] += 1
            pad_w = 1
        else:
            pad_w = 0
        shape = [shape[0] * 4, shape[1], int(shape[2] / 2), int(shape[3] / 2)]

    length = 8

    tensor = np.zeros(shape, dtype=int)
    num_groups = math.ceil(shape[1] / length)
    random16 = {}

    for group in range(num_groups):
        for offset in range(length):
            if group * length + offset >= shape[1]:
                break
            for rng_idx in range(shape[0] // 16):
                if offset == 0:
                    seed = (
                        ((layer + 1) << 10)
                        ^ ((rng_idx + 1) << 5)
                        ^ (group + 1)
                    )
                    random16[rng_idx] = xorshift_offset(
                        xor16_offset, init_seed=(seed,)
                    )

                out = random16[rng_idx]()
                for _l in range(16):
                    bit_value = int((out & (2**_l)) != 0)
                    tensor[rng_idx * 16 + _l][group * length + offset][0][
                        0
                    ] = bit_value

    if stride == 1:
        if flip is True:
            tensor = flip_odd(tensor)
        return tensor[: orig_shape[0], :, :, :]

    elif stride == 2:
        tensor = flip_odd(tensor)
        tensor = trans_st2_reverse(tensor)
        return tensor[: orig_shape[0], :, pad_h:, pad_w:]


def xorshift(generator, init_seed=None):
    ret = init_seed

    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator(*ret)
        return ret[-1]

    return inner


def xor16(y=1234):
    # print(y)
    y = y ^ (y << 7 & 0xFFFF)
    y = y ^ (y >> 9 & 0xFFFF)
    y = y ^ (y << 8 & 0xFFFF)
    return (y & 0xFFFF,)


def maketensor_xor16_ko(shape, layer=1, stride=1, flip=False, args=None):
    assert stride < 3, "not implemented error!"

    orig_shape = shape

    tmp_shape = []
    for i in range(len(shape)):
        tmp_shape.append(shape[i])
    shape = tmp_shape
    bchannel = 16 if stride == 1 else 4
    shape[0] = math.ceil(shape[0] / bchannel) * bchannel
    if stride == 2:
        if shape[2] % 2 == 1:
            shape[2] += 1
            pad_h = 1
        else:
            pad_h = 0
        if shape[3] % 2 == 1:
            shape[3] += 1
            pad_w = 1
        else:
            pad_w = 0
        shape = [shape[0] * 4, shape[1], int(shape[2] / 2), int(shape[3] / 2)]

    tensor = np.zeros(shape, dtype=int)
    random16 = {}

    for i in range(shape[1]):
        for j in range(shape[0] // 16):
            if i == 0:
                seed = ((layer + 1) << 10) ^ (j + 1)
                random16[j] = xorshift(xor16, init_seed=(seed,))
            for k in range(shape[2] * shape[3]):
                out = random16[j]()
                for _l in range(16):
                    bit_value = int((out & (2**_l)) != 0)
                    tensor[j * 16 + _l][i][k // shape[2]][
                        k % shape[3]
                    ] = bit_value

    if stride == 1:
        if flip is True:
            tensor = flip_odd(tensor)
        return tensor[: orig_shape[0], :, :, :]

    elif stride == 2:
        tensor = flip_odd(tensor)
        tensor = trans_st2_reverse(tensor)
        return tensor[: orig_shape[0], :, pad_h:, pad_w:]


def xorshift_flowgnn_debug(generator, seed=None):
    ret = seed

    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator(*ret)
        return ret[-1]

    return inner


def xor16_flowgnn_debug(y=1234):
    # print(y)
    y = y ^ (y << 7 & 0xFFFF)
    y = y ^ (y >> 9 & 0xFFFF)
    y = y ^ (y << 8 & 0xFFFF)
    return (y & 0xFFFF,)


def maketensor_xor16_ko_flowgnn_debug(
    shape, layer=1, stride=1, flip=False, args=None
):
    assert stride < 3, "not implemented error!"

    # To avoid layer=0
    layer += 1
    orig_shape = shape

    tmp_shape = []
    for i in range(len(shape)):
        tmp_shape.append(shape[i])
    shape = tmp_shape
    bchannel = 16 if stride == 1 else 4
    shape[0] = math.ceil(shape[0] / bchannel) * bchannel
    if stride == 2:
        if shape[2] % 2 == 1:
            shape[2] += 1
            pad_h = 1
        else:
            pad_h = 0
        if shape[3] % 2 == 1:
            shape[3] += 1
            pad_w = 1
        else:
            pad_w = 0
        shape = [shape[0] * 4, shape[1], int(shape[2] / 2), int(shape[3] / 2)]

    tensor = np.zeros(shape)
    seeds_matrix = np.zeros((shape[1], int(shape[0] / 16)), dtype=np.uint16)
    seeds_vector = np.zeros(shape[1] * int(shape[0] / 16), dtype=np.uint16)

    # if args is not None and args.xor_seed_using_instance_number:
    #     for i in range(shape[1]):
    #         for j in range(int(shape[0] / 16)):
    #             # seed = (layer << 10) ^ i ^ (j << 5)
    #             seed = (layer << 13) ^ (i << 6) ^ j
    #             random16 = xorshift(xor16, seed=(seed,))
    #             for k in range(shape[2] * shape[3]):
    #                 out = random16()
    #                 seeds_matrix[i, j] = out
    #                 seeds_vector[i * int(shape[0] / 16) + j] = out
    #                 for _l in range(16):
    #                     tensor[j * 16 + _l][i][k // shape[2]][k % shape[3]] = int((out & (2**_l)) != 0)
    # else:
    for i in range(shape[1]):
        # layer = 1
        # i = 267
        seed = (layer << 10) ^ i
        random16 = xorshift_flowgnn_debug(xor16_flowgnn_debug, seed=(seed,))
        for j in range(int(shape[0] / 16)):
            for k in range(shape[2] * shape[3]):
                out = random16()
                # print(f"layer:{layer}, i:{i}, out:{out}")
                seeds_matrix[i, j] = out
                seeds_vector[i * int(shape[0] / 16) + j] = out
                for _l in range(16):
                    tensor[j * 16 + _l][i][k // shape[2]][k % shape[3]] = int(
                        (out & (2**_l)) != 0
                    )

    # np.savetxt("seeds_matrix.txt", seeds_matrix, fmt="%d")
    # with open("seeds_matrix.bin", "wb") as f:
    #     f.write(seeds_matrix.tobytes())

    # np.savetxt("seeds_vector.txt", seeds_vector, fmt="%d")
    # with open("seeds_vector.bin", "wb") as f:
    #     f.write(seeds_vector.tobytes())

    if stride == 1:
        if flip is True:
            tensor = flip_odd(tensor)
        return tensor[: orig_shape[0], :, :, :]
    elif stride == 2:
        tensor = flip_odd(tensor)
        tensor = trans_st2_reverse(tensor)
        return tensor[: orig_shape[0], :, pad_h:, pad_w:]


def xorshift256(generator, init_seed=None):
    ret = init_seed

    def inner():
        nonlocal ret
        if ret is None:
            ret = generator()
        else:
            ret = generator()
        return ret

    return inner


class Xoshiro256pp:
    def __init__(self, init_seed):
        self.state = self.init_seed_state(init_seed)

    @staticmethod
    def rotl(x, k):
        return np.uint64(
            (np.uint64(x) << np.uint64(k))
            | (np.uint64(x) >> np.uint64(64 - k))
        )

    @staticmethod
    def init_seed_state(init_seed):
        state = [0] * 4
        for i in range(4):
            init_seed = np.bitwise_xor(
                np.bitwise_xor(
                    np.bitwise_xor(
                        np.uint64(init_seed),
                        np.left_shift(
                            np.uint64(init_seed), 13, dtype=np.uint64
                        ),
                    ),
                    np.right_shift(np.uint64(init_seed), 7, dtype=np.uint64),
                ),
                np.left_shift(np.uint64(init_seed), 17, dtype=np.uint64),
            )
            state[i] = init_seed
        return state

    # def init_seed_state(init_seed):
    #     # This is a simple implementation using splitmix64 to generate initial state
    #     def splitmix64(init_seed):
    #         z = np.uint64(init_seed) + np.uint64(0x9E3779B97F4A7C15)
    #         z = np.uint64(z ^ (z >> np.uint64(30))) * np.uint64(
    #             0xBF58476D1CE4E5B9
    #         )
    #         z = np.uint64(z ^ (z >> np.uint64(27))) * np.uint64(
    #             0x94D049BB133111EB
    #         )
    #         return np.uint64(z ^ (z >> np.uint64(31)))

    #     state = [0] * 4
    #     for i in range(4):
    #         init_seed = splitmix64(init_seed)
    #         state[i] = init_seed
    #     return state

    def next(self):
        result = self.rotl(
            np.uint64(self.state[0] + self.state[3]), 23
        ) + np.uint64(self.state[0])
        t = np.uint64(self.state[1] << np.uint64(17))

        self.state[2] = np.uint64(self.state[2] ^ self.state[0])
        self.state[3] = np.uint64(self.state[3] ^ self.state[1])
        self.state[1] = np.uint64(self.state[1] ^ self.state[2])
        self.state[0] = np.uint64(self.state[0] ^ self.state[3])

        self.state[2] = np.uint64(self.state[2] ^ t)
        self.state[3] = self.rotl(self.state[3], 45)

        return result

    def next256(self):
        return [self.next() for _ in range(4)]  # Generate 4 64-bit values


def maketensor_xor256(shape, layer=0, stride=1, flip=False, method="same"):
    assert stride < 3, "Stride not implemented error!"

    orig_shape = shape.clone()  # 修正箇所

    bchannel = 256  # Adjusted bchannel to 256 for simplicity
    shape = shape.numpy()  # 修正箇所: TensorをNumpy配列に変換
    shape[0] = int(np.ceil(shape[0] / bchannel) * bchannel)

    tensor = np.zeros(shape, dtype=np.uint8)
    init_seeds_matrix = np.zeros(
        (shape[1], int(shape[0] / 256), 4), dtype=np.uint64
    )  # init_seeds_matrixを追加

    for i in range(shape[1]):
        init_seed = np.uint64(((layer + 1) << 10) ^ (i + 1))
        random256 = xorshift256(
            Xoshiro256pp(init_seed).next256, init_seed=(init_seed,)
        )
        for j in range(int(shape[0] / 256)):
            out = random256()
            init_seeds_matrix[i, j, :] = out  # init_seeds_matrixに乱数を保存
            for k in range(4):
                bit_array = np.unpackbits(
                    np.array([out[k]], dtype=np.uint64).view(np.uint8)
                )
                tensor[j * 256 + k * 64 : j * 256 + (k + 1) * 64, i, :, :] = (
                    bit_array.reshape((64, 1, 1))
                )

    # np.savetxt(
    #     f"init_seeds_matrix_L{layer}_{shape[1]}_{shape[0]//64}.txt",
    #     init_seeds_matrix.reshape(-1, 4 * int(np.ceil(shape[0] / bchannel))),
    #     fmt="%d",
    # )
    # with open("init_seeds_matrix.bin", "wb") as f:
    #     f.write(init_seeds_matrix.tobytes())

    if stride == 1:
        if flip:
            tensor = np.flip(
                tensor, axis=[2]
            )  # Assuming flipping along a specific axis
        return tensor[
            : orig_shape[0].item(), :, :, :
        ]  # 修正箇所: orig_shape[0]を整数に変換
    elif stride == 2:
        raise NotImplementedError("Stride of 2 is not implemented.")


def sparse_mul_counter(adj, x):
    if len(adj.shape) == 3:
        adj = torch.squeeze(adj, 0)
    if len(x.shape) == 3:
        x = torch.squeeze(x, 0)
    sum = torch.spmm(adj, x)
    non_zero = torch.count_nonzero(sum)
    adj_sparse = adj.to_sparse()
    x_sparse = x.to_sparse()

    n_mul_tot = 0  # 非零乘法次数

    # 遍历adj的每一行
    for i in range(adj_sparse.shape[0]):
        row_indices = adj_sparse.indices()[0] == i  # 找到第i行的非零元素的索引
        cols = adj_sparse.indices()[1][row_indices]  # 对应的列索引

        # 对于每一列，计算与x中相应列的非零元素乘法次数
        for col in cols:
            n_mul_tot += x_sparse.indices()[1][
                x_sparse.indices()[0] == col
            ].size(0)
    MACs = n_mul_tot + n_mul_tot - non_zero
    return MACs.item()


def concrete_neuron(logit_p, train=False, temp=1.0, **kwargs):
    """
    Use concrete distribution to approximate binary output. Here input is logit(keep_prob).
    """
    if train is False:
        result = logit_p.data.new().resize_as_(logit_p.data).fill_(1.0)
        result[logit_p.data < 0.0] = 0.0
        return result

    # Note that p is the retain probability here
    p = torch.sigmoid(logit_p)
    unif_noise = logit_p.data.new().resize_as_(logit_p.data).uniform_()

    approx = (
        torch.log(1.0 - p + EPSILON)
        - torch.log(p + EPSILON)
        + torch.log(unif_noise + EPSILON)
        - torch.log(1.0 - unif_noise + EPSILON)
    )
    drop_prob = torch.sigmoid(approx / temp)
    keep_prob = 1.0 - drop_prob
    mask = keep_prob.clone().detach() > 0.5
    # mask=p.clone().detach()>0.5
    mask = mask.float()
    out = mask - keep_prob.detach() + keep_prob
    return out


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        # k_val = percentile(scores, sparsity*100)
        # if glob:
        out = torch.where(
            scores < threshold, zeros.to(scores.device), ones.to(scores.device)
        )
        # else:
        #    k_val = percentile(scores, threshold*100)
        #    out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def percentile(t, q):
    t_flat = t.view(-1)
    t_sorted, _ = torch.sort(t_flat)
    k = 1 + round(0.01 * float(q) * (t.numel() - 1)) - 1
    return t_sorted[k].item()


class SparseModule(nn.Module):
    def init_param_(
        self,
        param,
        init_mode=None,
        scale=None,
        sparse_value=None,
        layer=None,
        list_num=None,
    ):
        if init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == "uniform":
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == "kaiming_normal_SF":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            param.data.normal_(0, std)
        elif init_mode == "signed_constant":
            # From github.com/allenai/hidden-networks
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        elif (
            init_mode == "signed_constant_sparse"
            or init_mode == "signed_constant_SF"
        ):
            # From okoshi'san's M-sup paper
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        elif init_mode == "xor16_flowgnn_debug":
            # From okoshi'san's M-sup paper
            shape = torch.tensor(param.size())
            new_value = torch.tensor([1, 1])
            shape = torch.cat((shape, new_value), dim=0)
            a = torch.tensor(
                maketensor_xor16_ko_flowgnn_debug(
                    shape, layer=layer, stride=1, flip=False
                ),
                dtype=torch.float32,
            )
            a_reshaped = a.squeeze()
            a_sign = a_reshaped * 2 - 1  # 将0变为-1，1保持不变
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            # nn.init.kaiming_normal_(param)    # use only its sign
            param.data = a_sign * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        elif init_mode == "xor16":
            # From okoshi'san's M-sup paper
            shape = torch.tensor(param.size())
            new_value = torch.tensor([1, 1])
            shape = torch.cat((shape, new_value), dim=0)
            a = torch.tensor(
                maketensor_xor16_ko(shape, layer=layer, stride=1, flip=False),
                dtype=torch.float32,
            )
            a_reshaped = a.squeeze()
            a_sign = a_reshaped * 2 - 1  # 将0变为-1，1保持不变
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            # nn.init.kaiming_normal_(param)    # use only its sign
            param.data = a_sign * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        elif init_mode == "xor16_offset":
            # From okoshi'san's M-sup paper
            shape = torch.tensor(param.size())
            new_value = torch.tensor([1, 1])
            shape = torch.cat((shape, new_value), dim=0)
            a = torch.tensor(
                maketensor_xor16_ko_offset(
                    shape, layer=layer, stride=1, flip=False
                ),
                dtype=torch.float32,
            )
            a_reshaped = a.squeeze()
            a_sign = a_reshaped * 2 - 1  # 将0变为-1，1保持不变
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            # nn.init.kaiming_normal_(param)    # use only its sign
            param.data = a_sign * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        elif init_mode == "xor16_atom":
            param = param.transpose(0, 1)
            original_shape = param.size()
            # From okoshi'san's M-sup paper
            shape = torch.tensor(original_shape)
            new_value = torch.tensor([1, 1])
            shape = torch.cat((shape, new_value), dim=0)
            a = torch.tensor(
                maketensor_xor16_ko_atom(
                    shape, layer=layer, stride=1, flip=False
                ),
                dtype=torch.float32,
            )
            a_reshaped = a.squeeze()
            a_sign = a_reshaped * 2 - 1  # 将0变为-1，1保持不变
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            param.copy_(a_sign * std * scale)  # paramを更新
            param = param.transpose(0, 1)
        elif init_mode == "xor16_bond":
            param = param.transpose(0, 1)
            original_shape = param.size()
            # From okoshi'san's M-sup paper
            shape = torch.tensor(original_shape)
            new_value = torch.tensor([1, 1])
            shape = torch.cat((shape, new_value), dim=0)
            a = torch.tensor(
                maketensor_xor16_ko_bond(
                    shape, layer=layer, stride=1, flip=False, list_num=list_num
                ),
                dtype=torch.float32,
            )
            a_reshaped = a.squeeze()
            a_sign = a_reshaped * 2 - 1  # 将0变为-1，1保持不变
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            param.copy_(a_sign * std * scale)  # paramを更新
            a_sign = a_sign.transpose(0, 1)
        elif init_mode == "xor256":
            # From okoshi'san's M-sup paper
            shape = torch.tensor(param.size())
            new_value = torch.tensor([1, 1])
            shape = torch.cat((shape, new_value), dim=0)
            a = torch.tensor(
                maketensor_xor256(shape, layer=layer, stride=1, flip=False),
                dtype=torch.float32,
            )
            a_reshaped = a.squeeze()
            a_sign = a_reshaped * 2 - 1  # 将0变为-1，1保持不变
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain("relu")
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            # nn.init.kaiming_normal_(param)    # use only its sign
            param.data = a_sign * std
            param.data *= scale  # scale value is defined in defualt as 1.0
            param = param.transpose(0, 1)
        else:
            raise NotImplementedError

    def rerandomize_(
        self,
        param,
        mask,
        mode=None,
        la=None,
        mu=None,
        init_mode=None,
        scale=None,
        param_twin=None,
        param_score=None,
    ):
        if param_twin is None:
            raise NotImplementedError
        else:
            param_twin = param_twin.to(param.device)

        with torch.no_grad():
            if mode == "bernoulli":
                assert (la is not None) and (mu is None)
                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                ones = torch.ones(param.size()).to(param.device)
                b = torch.bernoulli(ones * la)

                t1 = param.data * mask
                t2 = param.data * (1 - mask) * (1 - b)
                t3 = rnd.data * (1 - mask) * b

                param.data = t1 + t2 + t3
                # for score
                # b1=torch.bernoulli(ones*la*0.5)
                # param_score.data=param_score.data*mask
            elif mode == "manual":
                assert (la is not None) and (mu is not None)

                t1 = param.data * (1 - mask)
                t2 = param.data * mask

                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                rnd *= 1 - mask

                param.data = (t1 * la + rnd.data * mu) + t2
            else:
                raise NotImplementedError


"""
class SparseConv2d(SparseModule):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.bias_flag = kwargs['bias'] if 'bias' in kwargs else True
        self.padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs else None

        cfg = kwargs['cfg']
        self.sparsity = cfg['conv_sparsity']
        self.init_mode = cfg['init_mode']
        self.init_mode_mask = cfg['init_mode_mask']
        self.init_scale = cfg['init_scale']
        self.init_scale_score = cfg['init_scale_score']
        self.rerand_rate = cfg['rerand_rate']
        self.function = F.conv2d

        self.initialize_weights(2)

    def initialize_weights(self, convdim=None):
        if convdim == 1:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size))
        elif convdim == 2:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size))
        else:
            raise NotImplementedError

        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(self.out_ch))
        else:
            self.bias = None

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score) # noqa
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
    def get_subnet(self, weight_score=None):
        if weight_score is None:
            weight_score = self.weight_score

        subnet = GetSubnet.apply(self.weight_score, self.sparsity,
                                 self.weight_zeros, self.weight_ones)
        return subnet
    def forward(self, input):
        subnet = self.get_subnet(self.weight_score)
        pruned_weight = self.weight * subnet
        ret = self.function(
            input, pruned_weight, self.bias, self.stride, self.padding,
        )
        return ret
    def rerandomize(self, mode, la, mu):
        rate = self.rerand_rate
        mask = GetSubnet.apply(self.weight_score, self.sparsity * rate,
                               self.weight_zeros, self.weight_ones)
        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin)
class SparseConv1d(SparseConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = F.conv1d
        self.initialize_weights(1)
"""


# def get_threshold_local(sparsity, weight_score):
#     if self.args.enable_mask is True:  # enable multi-mask
#         threshold_list = []
#         for value in sparsity:
#             local = weight_score.detach().flatten()
#             if self.args.enable_abs_comp is False:
#                 threshold = percentile(local, value * 100)
#             else:
#                 threshold = percentile(local.abs(), value * 100)
#             threshold_list.append(threshold)
#         return threshold_list
#     else:
#         local = weight_score.detach().flatten()
#         if self.args.enable_abs_comp is False:
#             threshold = percentile(local, sparsity * 100)
#         else:
#             threshold = percentile(local.abs(), sparsity * 100)
#         return threshold


class SparseLinear(SparseModule):
    def __init__(self, in_ch, out_ch, bias=False, layer=0, args=None):
        super().__init__()

        self.args = args

        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity

        if args.init_mode_linear is not None:
            self.init_mode = args.init_mode_linear
        else:
            self.init_mode = args.init_mode

        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score
        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.layer = layer
        self.folded_layer = layer
        self.enable_sw_mm = args.enable_sw_mm
        self.sparsity_value = args.linear_sparsity

        self.regular_weight_pruning = args.regular_weight_pruning
        self.num_of_weight_blocks = args.num_of_weight_blocks

        if self.regular_weight_pruning == "block":

            self.weight_scores = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(out_ch, 1))
                    for _ in range(args.num_of_weight_blocks)
                ]
            )
            for weight_score in self.weight_scores:
                weight_score.is_score = True
                weight_score.is_weight_score = True
                weight_score.sparsity = self.sparsity

            self.bias = None
            # self.weight_twin = torch.zeros(1, self.weight.size()[1])
            # self.weight_twin.requires_grad = False
            self.weight_zeros = torch.zeros(out_ch, 1)
            self.weight_ones = torch.ones(out_ch, 1)
            self.weight_zeros.requires_grad = False
            self.weight_ones.requires_grad = False

        elif self.regular_weight_pruning == "width":

            self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
            self.weight_score.is_score = True
            self.weight_score.is_weight_score = True
            self.weight_score.sparsity = self.sparsity
            self.bias = None
            self.weight_zeros = torch.zeros(out_ch, 1)
            self.weight_ones = torch.ones(out_ch, 1)
            self.weight_zeros.requires_grad = False
            self.weight_ones.requires_grad = False

        else:
            if self.enable_sw_mm is True:
                self.weight_scores = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(self.weight.size()))
                        for _ in range(self.folded_layer)
                    ]
                )
                for weight_score in self.weight_scores:
                    weight_score.is_score = True
                    weight_score.is_weight_score = True
                    weight_score.sparsity = self.sparsity
            else:
                self.weight_score = nn.Parameter(
                    torch.ones(self.weight.size())
                )
                self.weight_score.is_score = True
                if args.train_mode == "score_only":
                    self.weight_score.is_weight_score = True
                self.weight_score.sparsity = self.sparsity

            if self.args.train_mode == "normal":
                self.bias = nn.Parameter(torch.zeros(out_ch))
                # self.bias.sparsity = self.sparsity
                # self.bias.is_score = True
                # weight_score.is_weight_score = True
                # self.bias.sparsity = 0.0
            else:
                self.bias = None

            self.weight_zeros = torch.zeros(self.weight.size())
            self.weight_ones = torch.ones(self.weight.size())
            self.weight_zeros.requires_grad = False
            self.weight_ones.requires_grad = False

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        if self.enable_sw_mm is True or self.regular_weight_pruning == "block":
            for weight_score in self.weight_scores:
                self.init_param_(
                    weight_score,
                    init_mode=self.init_mode_mask,
                    scale=self.init_scale_score,
                    layer=layer,
                )
        else:
            self.init_param_(
                self.weight_score,
                init_mode=self.init_mode_mask,
                scale=self.init_scale_score,
                layer=layer,
            )

        self.init_param_(
            self.weight,
            init_mode=self.init_mode,
            scale=self.init_scale,
            sparse_value=self.sparsity_value,
            layer=layer,
        )

        # self.weight_zeros = torch.zeros(self.weight_score.size())
        # self.weight_ones = torch.ones(self.weight_score.size())

        # test
        # self.weight.requires_grad = False
        self.download_prunedw = args.download_prunedw
        self.prunedw_path = args.prunedw_path
        self.enable_abs_comp = args.enable_abs_comp
        self.args = args

        if args.train_mode == "score_only":
            self.weight.requires_grad = False

        self.nmsparsity = args.nmsparsity
        # self.N = args.N
        # self.M = args.M
        self.decay = args.weight_decay

    def get_threshold_local(self, sparsity, weight_score):
        if self.args.enable_mask is True:  # enable multi-mask
            threshold_list = []
            for value in sparsity:
                local = weight_score.detach().flatten()
                if self.args.enable_abs_comp is False:
                    threshold = percentile(local, value * 100)
                else:
                    threshold = percentile(local.abs(), value * 100)
                threshold_list.append(threshold)
            return threshold_list
        else:
            local = weight_score.detach().flatten()
            if self.args.enable_abs_comp is False:
                threshold = percentile(local, sparsity * 100)
            else:
                threshold = percentile(local.abs(), sparsity * 100)
            return threshold

    def forward(
        self, x, threshold, sparsity=None, manual_mask=None, index_mask=0
    ):
        if self.enable_sw_mm is True or self.regular_weight_pruning == "block":
            weight_scores = self.weight_scores
        else:
            weight_score = self.weight_score

        if self.args.regular_weight_pruning == "width":
            self.sparsity = sparsity

        flag_lable = False
        if (
            self.enable_sw_mm is True
            and weight_scores[0].sparsity != self.sparsity
        ):
            flag_lable = True
        elif self.regular_weight_pruning is not None:
            flag_lable = True
        elif (
            self.enable_sw_mm is False
            and weight_score.sparsity != self.sparsity
        ):
            flag_lable = True
        elif (
            self.enable_sw_mm is False
            and weight_score.sparsity != self.sparsity
        ):
            flag_lable = True
        else:
            flag_lable = False

        if flag_lable is True:

            if self.args.regular_weight_pruning is None:
                if self.enable_abs_comp is False:
                    if self.enable_sw_mm is True:
                        weight_score = self.weight_scores[index_mask]
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )
                    else:
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                else:
                    if self.enable_sw_mm is True:
                        weight_score = self.weight_scores[index_mask]
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )
                    else:
                        threshold = self.get_threshold_local(
                            self.sparsity, weight_score
                        )
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    # after_sparsity = calculate_sparsity(pruned_weight)
                    # print(pruned_weight.size())
                    # print(f"after: {after_sparsity:.8f}")

            elif self.regular_weight_pruning == "block":

                weight_shape = self.weight.shape
                split_size = weight_shape[1] // self.num_of_weight_blocks

                expanded_subnets = []

                for i in range(self.num_of_weight_blocks):
                    weight_score = self.weight_scores[i]
                    sparsity = self.sparsity
                    threshold = self.get_threshold_local(
                        sparsity, weight_score
                    )

                    subnet = GetSubnet.apply(
                        weight_score.abs(),
                        threshold,
                        self.weight_zeros,
                        self.weight_ones,
                    )
                    if i < self.num_of_weight_blocks - 1:
                        subnet_expanded = subnet.repeat(1, split_size)
                    else:
                        remaining_size = (
                            weight_shape[1]
                            - (self.num_of_weight_blocks - 1) * split_size
                        )
                        subnet_expanded = subnet.repeat(1, remaining_size)

                    expanded_subnets.append(subnet_expanded)

                subnet = torch.cat(expanded_subnets, dim=1)

            elif self.regular_weight_pruning == "width":

                weight_shape = self.weight.shape

                expanded_subnets = []

                for i in range(weight_shape[1]):
                    weight_score = self.weight_scores[i]
                    sparsity = self.sparsity

                    weight_score_i = self.weight_score[:, i].unsqueeze(1)
                    threshold_i = self.get_threshold_local(
                        self.sparsity, weight_score_i
                    )

                    subnet = GetSubnet.apply(
                        weight_score.abs(),
                        threshold,
                        self.weight_zeros,
                        self.weight_ones,
                    )

                    if self.args.global_th_for_rowbyrow:
                        subnet = GetSubnet.apply(
                            weight_score_i.abs(),
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    else:
                        subnet = GetSubnet.apply(
                            weight_score_i.abs(),
                            threshold_i,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    # if i < self.num_of_weight_blocks - 1:
                    # subnet_expanded = subnet.repeat(1, split_size)
                    # else:
                    #     remaining_size = (
                    #         weight_shape[1] - (self.num_of_weight_blocks - 1) * split_size
                    #     )
                    #     subnet_expanded = subnet.repeat(1, remaining_size)

                    expanded_subnets.append(subnet)

                subnet = torch.cat(expanded_subnets, dim=1)

            pruned_weight = self.weight * subnet

        else:
            if manual_mask is None:
                if self.enable_abs_comp is False:
                    if self.enable_sw_mm is True:
                        weight_score = self.weight_scores[index_mask]
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    elif self.regular_weight_pruning is not None:
                        for i in range(self.num_of_weight_blocks):
                            weight_score = self.weight_scores[i]
                            sparsity = self.sparsity
                            # threshold = get_threshold_local(sparsity, weight_score)
                            subnet = GetSubnet.apply(
                                weight_score.abs(),
                                threshold,
                                self.weight_zeros,
                                self.weight_ones,
                            )

                            weight_shape = self.weight.shape
                            split_size = (
                                weight_shape[1] // self.num_of_weight_blocks
                            )

                            regular_subnets = []

                            for i in range(self.num_of_weight_blocks - 1):
                                weight_block = self.weight[
                                    i * split_size : (i + 1) * split_size
                                ]
                                weight_block_subnet = weight_block * subnet
                                regular_subnets.append(weight_block_subnet)

                            weight_block = self.weight[
                                (self.num_of_weight_blocks - 1) * split_size :
                            ]
                            weight_block_subnet = weight_block * subnet
                            regular_subnets.append(weight_block_subnet)

                        pruned_weight = torch.cat(regular_subnets, dim=1)

                    else:
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    if self.regular_weight_pruning is None:
                        pruned_weight = self.weight * subnet
                else:
                    if self.enable_sw_mm is True:
                        weight_score = self.weight_scores[index_mask]
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    elif self.regular_weight_pruning is not None:
                        for i in range(self.num_of_weight_blocks):
                            weight_score = self.weight_scores[i]
                            sparsity = self.sparsity
                            # threshold = get_threshold_local(sparsity, weight_score)
                            subnet = GetSubnet.apply(
                                weight_score.abs(),
                                threshold,
                                self.weight_zeros,
                                self.weight_ones,
                            )

                            weight_shape = self.weight.shape
                            split_size = (
                                weight_shape[1] // self.num_of_weight_blocks
                            )

                            regular_subnets = []

                            for i in range(self.num_of_weight_blocks - 1):
                                weight_block = self.weight[
                                    i * split_size : (i + 1) * split_size
                                ]
                                weight_block_subnet = weight_block * subnet
                                regular_subnets.append(weight_block_subnet)

                            weight_block = self.weight[
                                (self.num_of_weight_blocks - 1) * split_size :
                            ]
                            weight_block_subnet = weight_block * subnet
                            regular_subnets.append(weight_block_subnet)

                        pruned_weight = torch.cat(regular_subnets, dim=1)

                    else:
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold,
                            self.weight_zeros,
                            self.weight_ones,
                        )

                    if self.regular_weight_pruning is None:
                        pruned_weight = self.weight * subnet
            else:
                pruned_weight = self.weight * manual_mask

        if self.args.evatime is True:
            execution_time = 0.0
            ave_time = 0.0
            for _ in range(self.args.evanum):
                start_time = time.time()
                ret = torch.matmul(x, pruned_weight.T)
                if self.bias is not None:
                    ret += self.bias
                end_time = time.time()
                execution_time = execution_time + end_time - start_time
            ave_time = execution_time / self.args.evanum * 1000
            Wsparsity = torch.sum(subnet == 0).item() / subnet.numel()
            Xsparsity = torch.sum(x == 0).item() / x.numel()
            print("For linear layer XW : X size is", x.size())
            print("X Sparsity is:", Xsparsity)
            print(
                "For linear layer XW: pruned_weight size is",
                pruned_weight.size(),
            )
            print("W Sparsity is :", Wsparsity)
            print("Execution time: ", ave_time, "millisecond")
            n_mul_tot = sparse_mul_counter(x, pruned_weight.T)
            print("XW's non-zero mul+add operatdfions is:", n_mul_tot)
        if self.args.evatime is True:
            ret = torch.matmul(x, pruned_weight.T)
            if self.bias is not None:
                ret += self.bias
        else:
            if self.regular_weight_pruning == "width":
                calculate_and_plot_sparsity(pruned_weight, self.args.exp_name)

            x = x.cuda()
            if self.download_prunedw:
                os.makedirs(
                    f"./pretrained_model/Models/WandB/l{self.layer+1}",
                    exist_ok=True,
                )
                weight_size = pruned_weight.size()
                size_str = "x".join(map(str, weight_size))
                base_path = f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_pruned_weight_{size_str}"

                # 辞書として保存
                # save_dict = {
                #     "pruned_weight": pruned_weight,  # テンソルを辞書に入れる
                #     # 必要に応じて他のデータも辞書に追加可能
                # }
                # torch.save(save_dict, f"{base_path}.pth")

                # バイナリ形式で重みを個別に保存
                # torch.save(pruned_weight, f"{base_path}.bin")

                pruned_weight_np = pruned_weight.detach().cpu().numpy()
                # pruned_weight = pruned_weight.astype(np.int32)
                with open(
                    f"{base_path}.bin",
                    "wb",
                ) as f:
                    f.write(pruned_weight_np.tobytes())

                np.savetxt(f"{base_path}.txt", pruned_weight_np)

                x = x.cuda()

            ret = F.linear(x, pruned_weight, self.bias)

            x = x.float()
            # ret = F.linear(x, pruned_weight, self.bias)

        if self.args.flowgnn_debug is True:
            os.makedirs(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/",
                exist_ok=True,
            )
            subnet = subnet.detach().cpu().numpy()
            subnet = subnet.astype(np.int32)
            with open(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}.bin",
                "wb",
            ) as f:
                f.write(subnet.tobytes())
            np.savetxt(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}.txt",
                subnet,
                delimiter=",",
                fmt="%d",
            )

            # FLATTEN
            flattened_subnet = subnet.ravel(order="F")

            output_txt_path = f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}_flatten.txt"
            np.savetxt(
                output_txt_path,
                flattened_subnet.reshape(-1, 1),
                fmt="%d",
                delimiter=",",
            )
            # flattened_subnet.tofile(f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}_flatten.bin")

            # 行数インデックス配列を生成
            row_indices = np.arange(0, subnet.shape[0]).reshape(-1, 1)

            # 各列に対して行数インデックスを繰り返す
            subnet_index = np.tile(row_indices, (1, subnet.shape[1]))
            subnet_index = subnet_index.astype(np.int32)
            with open(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}_index.bin",
                "wb",
            ) as f:
                f.write(subnet_index.tobytes())
            np.savetxt(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}_index.txt",
                subnet_index,
                delimiter=",",
                fmt="%d",
            )
            # FLATTEN
            flattened_subnet = subnet_index.ravel(order="F")
            output_txt_path = f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_subnet_{subnet.shape}_index_flatten.txt"
            np.savetxt(
                output_txt_path,
                flattened_subnet.reshape(-1, 1),
                fmt="%d",
                delimiter=",",
            )

            weight = self.weight.detach().cpu().numpy()

            # Save as binary file
            with open(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_weight_{self.weight.shape}.bin",
                "wb",
            ) as f:
                f.write(weight.tobytes())

            weight = weight.ravel(order="F")
            # Save as text file
            np.savetxt(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_weight_{self.weight.shape}.txt",
                weight,
                delimiter=",",
                fmt="%.6f",
            )

            bias = self.bias.detach().cpu().numpy()

            # Save as binary file
            with open(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_bias.bin",
                "wb",
            ) as f:
                f.write(bias.tobytes())

            # Save as text file
            np.savetxt(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_bias.txt",
                bias,
                delimiter=",",
                fmt="%.6f",
            )

        # calculate_and_plot_sparsity(pruned_weight, self.args.exp_name)

        return ret

    def rerandomize(self, mode, la, mu, sparsity, manual_mask=None):
        if sparsity is None:
            sparsity = self.sparsity
        if manual_mask is None:
            if self.enable_abs_comp is False:
                rate = self.rerand_rate
                mask = GetSubnet.apply(
                    torch.sigmoid(self.weight_score),
                    sparsity * rate,
                    self.weight_zeros,
                    self.weight_ones,
                )
            else:
                rate = self.rerand_rate
                mask = GetSubnet.apply(
                    torch.sigmoid(self.weight_score.abs()),
                    sparsity * rate,
                    self.weight_zeros,
                    self.weight_ones,
                )
        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(
            self.weight,
            mask,
            mode,
            la,
            mu,
            self.init_mode,
            scale,
            self.weight_twin,
            param_score=self.weight_score,
        )


class SparseLinearMulti_mask(SparseModule):
    def __init__(self, in_ch, out_ch, bias=False, layer=0, args=None):
        super().__init__()

        self.args = args
        self.sparsity = args.linear_sparsity
        self.init_mode = args.init_mode
        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))

        self.folded_layer = layer
        self.layer = layer
        self.enable_sw_mm = args.enable_sw_mm
        self.sparsity_value = args.linear_sparsity

        self.regular_weight_pruning = args.regular_weight_pruning
        self.num_of_weight_blocks = args.num_of_weight_blocks

        #         if self.regular_weight_pruning == "block":
        #
        #             self.weight_scores = nn.ParameterList(
        #                 [
        #                     nn.Parameter(torch.ones(out_ch, 1))
        #                     for _ in range(args.num_of_weight_blocks)
        #                 ]
        #             )
        #             for weight_score in self.weight_scores:
        #                 weight_score.is_score = True
        #                 weight_score.is_weight_score = True
        #                 weight_score.sparsity = self.sparsity
        #
        #             self.bias = None
        #             # self.weight_twin = torch.zeros(1, self.weight.size()[1])
        #             # self.weight_twin.requires_grad = False
        #             self.weight_zeros = torch.zeros(out_ch, 1)
        #             self.weight_ones = torch.ones(out_ch, 1)
        #             self.weight_zeros.requires_grad = False
        #             self.weight_ones.requires_grad = False
        #
        #         elif self.regular_weight_pruning == "width":
        #
        #             self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        #             self.weight_score.is_score = True
        #             self.weight_score.is_weight_score = True
        #             self.weight_score.sparsity = self.sparsity
        #             self.bias = None
        #             self.weight_zeros = torch.zeros(out_ch, 1)
        #             self.weight_ones = torch.ones(out_ch, 1)
        #             self.weight_zeros.requires_grad = False
        #             self.weight_ones.requires_grad = False
        #
        #             num_zeros_or = int(out_ch * (self.sparsity_value * 1.1))
        #             num_ones_or = out_ch - num_zeros_or
        #             zeros_or = torch.zeros(num_zeros_or, 1)
        #             ones_or = torch.ones(num_ones_or, 1)
        #             combined_or = torch.cat((zeros_or, ones_or), dim=0)
        #             permuted_indices_or = torch.randperm(out_ch)
        #             self.or_mask = combined_or[permuted_indices_or]
        #
        #             # and_maskを作成するためにself.sparsity_value + 0.05を使用
        #             num_zeros_and = int(out_ch * (self.sparsity_value * 0.9))
        #             num_ones_and = out_ch - num_zeros_and
        #             zeros_and = torch.zeros(num_zeros_and, 1)
        #             ones_and = torch.ones(num_ones_and, 1)
        #             combined_and = torch.cat((zeros_and, ones_and), dim=0)
        #             permuted_indices_and = torch.randperm(out_ch)
        #             self.and_mask = combined_and[permuted_indices_and]
        #
        #         else:
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.is_weight_score = True
        self.weight_score.sparsity = self.sparsity
        # self.weight_twin = torch.zeros(1, self.weight.size()[1])
        # self.weight_twin.requires_grad = False
        self.weight_zeros = torch.zeros(out_ch, 1)
        self.weight_ones = torch.ones(out_ch, 1)
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False

        self.bias = None

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_mask,
            scale=self.init_scale_score,
            layer=layer,
        )

        self.init_param_(
            self.weight,
            init_mode=self.init_mode,
            scale=self.init_scale,
            sparse_value=self.sparsity_value,
            layer=layer,
        )

        self.sp_list = args.sparsity_list
        self.en_mask = args.enable_mask
        self.enable_abs_comp = args.enable_abs_comp
        self.download_prunedw = args.download_prunedw
        self.prunedw_path = "./"
        self.weight.requires_grad = False
        self.args = args

    #
    #     def get_threshold_local(self, sparsity, weight_score):
    #         if self.args.enable_mask is True:  # enable multi-mask
    #             threshold_list = []
    #             for value in sparsity:
    #                 local = weight_score.detach().flatten()
    #                 if self.args.enable_abs_comp is False:
    #                     threshold = percentile(local, value * 100)
    #                 else:
    #                     threshold = percentile(local.abs(), value * 100)
    #                 threshold_list.append(threshold)
    #             return threshold_list
    #         else:
    #             local = weight_score.detach().flatten()
    #             if self.args.enable_abs_comp is False:
    #                 threshold = percentile(local, sparsity * 100)
    #             else:
    #                 threshold = percentile(local.abs(), sparsity * 100)
    #             return threshold
    #
    def forward(
        self, x, threshold, sparsity=None, manual_mask=None, index_mask=0
    ):
        subnets = []
        for threshold_v in threshold:
            subnet = GetSubnet.apply(
                (
                    torch.abs(self.weight_score)
                    if self.enable_abs_comp
                    else self.weight_score
                ),
                threshold_v,
                self.weight_zeros,
                self.weight_ones,
            )
            subnets.append(subnet)

        combined_subnet = torch.stack(subnets).sum(dim=0)
        pruned_weight = self.weight * combined_subnet

        ret = F.linear(x, pruned_weight, self.bias)
        return ret


#
#     def rerandomize(self, mode, la, mu, sparsity, manual_mask=None):
#         if sparsity is None:
#             sparsity = self.sparsity
#
#         for weight_score in self.weight_scores:
#             if manual_mask is None:
#                 rate = self.rerand_rate
#                 if self.enable_abs_comp is False:
#                     mask = GetSubnet.apply(
#                         torch.sigmoid(weight_score),
#                         sparsity * rate,
#                         self.weight_zeros,
#                         self.weight_ones,
#                     )
#                 else:
#                     mask = GetSubnet.apply(
#                         torch.sigmoid(weight_score.abs()),
#                         sparsity * rate,
#                         self.weight_zeros,
#                         self.weight_ones,
#                     )
#
#             else:
#                 mask = manual_mask
#
#             scale = self.init_scale
#             self.rerandomize_(
#                 self.weight,
#                 mask,
#                 mode,
#                 la,
#                 mu,
#                 self.init_mode,
#                 scale,
#                 self.weight_twin,
#                 param_score=weight_score,
#             )
#


class NMSparseLinear(SparseModule):
    def __init__(self, in_ch, out_ch, layer=0, args=None):
        super().__init__()

        self.args = args
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sparsity = args.linear_sparsity
        self.init_mode = args.init_mode
        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.M = args.M
        self.weight_score = nn.Parameter(torch.ones(in_ch, out_ch))
        self.bias = None

        self.init_param_(
            self.weight,
            init_mode=self.init_mode,
            scale=self.init_scale,
            sparse_value=self.args.linear_sparsity,
            layer=layer,
        )
        self.weight.requires_grad = False
        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_mask,
            scale=self.init_scale_score,
        )
        self.weight_score.is_score = True
        self.weight_score.is_weight_score = True
        self.weight_score.sparsity = self.sparsity

        self.nmsparsity = args.nmsparsity
        self.nm_decay = args.nm_decay
        # self.N = args.N

    def forward(self, x, threshold=None, sparsity=None):

        pruned_weight = GetNMSubnet.apply(
            self.weight, self.weight_score, self.M, sparsity, self.nm_decay
        )
        x = x.cuda().float()
        ret = F.linear(x, pruned_weight, self.bias)

        return ret


class NMSparseMultiLinear(SparseModule):
    def __init__(self, in_ch, out_ch, layer=0, args=None):
        super().__init__()

        self.args = args
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sparsity = args.linear_sparsity
        self.init_mode = args.init_mode
        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.M = args.M
        self.weight_score = nn.Parameter(torch.ones(in_ch, out_ch))
        self.bias = None
        self.layer = layer
        self.init_param_(
            self.weight,
            init_mode=self.init_mode,
            scale=self.init_scale,
            sparse_value=self.args.linear_sparsity,
            layer=layer,
        )
        self.weight.requires_grad = False
        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_mask,
            scale=self.init_scale_score,
        )
        self.weight_score.is_score = True
        self.weight_score.is_weight_score = True
        self.weight_score.sparsity = self.sparsity

        self.nmsparsity = args.nmsparsity
        self.nm_decay = args.nm_decay

    def forward(self, x, threshold=None, sparsity=None):
        pruned_weight, combined_subnet = GetNMMultiSubnet.apply(
            self.weight, self.weight_score, self.M, sparsity[0], self.nm_decay
        )
        # if self.args.flowgnn_debug is True:
        #     # test row_wise product
        #     row_wise_sum = torch.zeros(x.shape[0], pruned_weight.shape[0]).to(
        #         "cuda"
        #     )
        #     os.makedirs(
        #         "./pretrained_model/debug_txt",
        #         exist_ok=True,
        #     )
        #     for i in range(x.shape[0]):
        #         for j in range(x.shape[1]):
        #             if x[i][j] != 0:
        #                 row_wise_sum[i] += x[i][j] * pruned_weight[:, j]
        #                 with open(
        #                     f"./pretrained_model/debug_txt/row_wise_sum_{i}.txt",
        #                     "w",
        #                 ) as file:
        #                     for n in range(row_wise_sum.shape[1]):
        #                         file.write("%f\n" % row_wise_sum[i][n])
        #     print("computer 1 NZ")
        #     row_wise_sum[i] = torch.sum(x[i].unsqueeze(0) * pruned_weight, dim=1)
        #     with open(
        #         "./pretrained_model/debug_txt/row_wise_sum.txt",
        #         "w",
        #     ) as file:
        #         for i in range(row_wise_sum.shape[0]):
        #             for j in range(row_wise_sum.shape[1]):
        #                 file.write("%f\n" % row_wise_sum[i][j])

        ret = F.linear(x, pruned_weight, None)

        if self.args.flowgnn_debug is True:
            combined_subnet = combined_subnet.detach().cpu().numpy()
            os.makedirs(
                f"./pretrained_model/Models/WandB/l{self.layer+1}",
                exist_ok=True,
            )
            with open(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_combined_subnet_{combined_subnet.shape}.bin",
                "wb",
            ) as f:
                f.write(combined_subnet.tobytes())
            flattened_subnet = combined_subnet.ravel(order="F")
            output_txt_path = f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_combined_subnet_{combined_subnet.shape}.txt"
            np.savetxt(
                output_txt_path,
                flattened_subnet.reshape(-1, 1),
                fmt="%d",
                delimiter=",",
            )
            position_encoded, value_encoded = self.encode_array(
                combined_subnet
            )
            self.save_arrays(position_encoded, value_encoded)
            weight = self.weight.detach().cpu().numpy()
            result = np.zeros(weight.shape[0] * weight.shape[1], dtype=int)
            index = 0
            for i in range(weight.shape[1]):
                for j in range(weight.shape[0]):
                    if weight[j, i] > 0:
                        result[index] = 1
                    else:
                        result[index] = 0
                    index += 1
            with open(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_weight_{self.weight.shape}.bin",
                "wb",
            ) as f:
                f.write(weight.tobytes())
            weight = weight.ravel(order="F")
            np.savetxt(
                f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_weight_{self.weight.shape}.txt",
                weight,
                delimiter=",",
                fmt="%.6f",
            )

        return ret

    def encode_array(self, input_array):
        rows, cols = input_array.shape
        num_blocks = rows // 16  # 16行ごとにブロックを形成
        position_array = np.zeros((num_blocks * 8, cols), dtype=np.uint8)
        value_array = np.zeros((num_blocks * 8, cols), dtype=np.uint8)

        for col in range(cols):
            for block in range(num_blocks):
                start_idx = block * 16
                block_data = input_array[start_idx : start_idx + 16, col]

                pos_idx = 0
                for i, value in enumerate(block_data):
                    if value != 0:
                        position = i
                        encoded_value = value
                        position_array[pos_idx + block * 8, col] = position
                        value_array[pos_idx + block * 8, col] = encoded_value
                        pos_idx += 1

        return position_array, value_array

    def save_arrays(self, position_array, value_array):
        os.makedirs(
            f"./pretrained_model/Models/WandB/l{self.layer+1}",
            exist_ok=True,
        )
        position_array = position_array.astype(np.int32)
        value_array = value_array.astype(np.int32)
        # テキスト形式で保存
        # np.savetxt(
        #     f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_positions.txt",
        #     position_array,
        #     fmt="%d",
        #     delimiter=",",
        # )
        # np.savetxt(
        #     f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_values.txt",
        #     value_array,
        #     fmt="%d",
        #     delimiter=",",
        # )

        # # バイナリ形式で保存
        # position_array.tofile(
        #     f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_positions.bin"
        # )
        with open(
            f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_positions.bin",
            "wb",
        ) as f:
            f.write(position_array.tobytes())
        # value_array.tofile(
        #     f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_values.bin"
        # )
        with open(
            f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_values.bin",
            "wb",
        ) as f:
            f.write(value_array.tobytes())

        # FLATTEN
        flattened_position_array = position_array.ravel(order="F")
        flattened_value_array = value_array.ravel(order="F")
        np.savetxt(
            f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_positions.txt",
            flattened_position_array,
            fmt="%d",
            delimiter=",",
        )
        np.savetxt(
            f"./pretrained_model/Models/WandB/l{self.layer+1}/l{self.layer+1}_NM_values.txt",
            flattened_value_array,
            fmt="%d",
            delimiter=",",
        )

        # バイナリ形式で保存
        # flattened_position_array.tofile(f"./pretrained_model/s_layer{self.layer}_flatten.bin")
        # flattened_value_array.tofile(f"./pretrained_model/NM_values_layer{self.layer}_flatten.bin")


class GetNMSubnet(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, weight_score, N, M, decay=0.0015):
        ctx.save_for_backward(weight_score)

        num_rows, num_cols = weight_score.shape
        group_size = M
        num_full_groups = num_cols // group_size
        last_group_size = num_cols % group_size

        mask = torch.ones_like(weight_score)

        for group in range(num_full_groups):
            start_idx = group * group_size
            end_idx = (group + 1) * group_size
            block = weight_score[:, start_idx:end_idx].detach().abs()
            _, indices = torch.topk(
                block,
                min(int(group_size - N), block.size(1)),
                largest=False,
                dim=1,
            )
            mask[:, start_idx:end_idx].scatter_(1, indices, 0)

        if last_group_size > 0:
            start_idx = num_full_groups * group_size
            end_idx = num_cols
            block = weight_score[:, start_idx:end_idx].detach().abs()
            last_group_N = int(last_group_size * (1 - N / M))
            last_group_N = max(1, last_group_N)
            _, indices = torch.topk(
                block,
                int(last_group_size - last_group_N),
                largest=False,
                dim=1,
            )
            mask[:, start_idx:end_idx].scatter_(1, indices, 0)

        ctx.mask = mask
        ctx.decay = decay
        ctx.weight = weight

        # 行ごとの非ゼロ要素数をプリント
        nonzero_counts = mask.count_nonzero(dim=1)
        print("Number of nonzero elements per row:")
        for i, count in enumerate(nonzero_counts):
            print(f"Row {i}: {count.item()}")

        return weight * mask.T

    @staticmethod
    def backward(ctx, grad_output):

        (weight,) = ctx.saved_tensors
        return grad_output + ctx.decay * (1 - ctx.mask) * weight, None, None


class GetNMMultiSubnet(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, weight_score, M, sparsity, decay):
        # if torch.is_grad_enabled():
        #     ctx.save_for_backward(weight_score)
        ctx.save_for_backward(weight_score)

        num_rows, num_cols = weight_score.shape
        group_size = M
        num_full_groups = num_cols // group_size
        last_group_size = num_cols % group_size

        mask = torch.ones_like(weight_score)
        N = int(M * (1 - sparsity))

        for group in range(num_full_groups):

            start_idx = group * group_size
            end_idx = (group + 1) * group_size
            block = weight_score[:, start_idx:end_idx].detach().abs()
            _, indices = torch.topk(
                block,
                min(int(group_size - N), block.size(1)),
                largest=False,
                dim=1,
            )
            mask[:, start_idx:end_idx].scatter_(1, indices, 0)

        if last_group_size > 0:
            start_idx = num_full_groups * group_size
            end_idx = num_cols
            block = weight_score[:, start_idx:end_idx].detach().abs()
            last_group_N = int(last_group_size * (1 - sparsity))
            last_group_N = max(1, last_group_N)
            _, indices = torch.topk(
                block,
                int(last_group_size - last_group_N),
                largest=False,
                dim=1,
            )
            mask[:, start_idx:end_idx].scatter_(1, indices, 0)

        indices_of_ones = torch.nonzero(mask == 1, as_tuple=True)
        scores_of_ones = weight_score[indices_of_ones].detach().abs()
        threshold_low = scores_of_ones.quantile(0.33333)
        threshold_high = scores_of_ones.quantile(0.66666)

        mask[indices_of_ones] = torch.where(
            scores_of_ones < threshold_low,
            1.0,
            torch.where(scores_of_ones < threshold_high, 2, 3),
        )

        ctx.mask = mask
        ctx.decay = decay
        ctx.weight = weight

        return weight * mask.T, mask.T

    @staticmethod
    def backward(ctx, grad_output, grad_extra):
        return (
            None,
            grad_output.T - ctx.decay * (3 - ctx.mask) * ctx.weight.T,
            None,
            None,
            None,
        )
        # return None, grad_output.T + ctx.decay * (1 - ctx.mask) * ctx.weight.T, None, None


# class GetNMMultiSubnet(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, weight, weight_score, N, M, decay=0.0015):
#         ctx.save_for_backward(weight_score)

#         length = weight_score.numel()
#         group = int(length / M)

#         weight_score_temp = weight_score.detach().abs().reshape(group, M)
#         index = torch.argsort(weight_score_temp, dim=1)[:, : int(M - N)]

#         w_b = torch.ones(weight_score_temp.shape, device=weight_score_temp.device)
#         w_b = w_b.scatter_(dim=1, index=index, value=0)

#         indices_of_ones = torch.nonzero(w_b == 1, as_tuple=True)

#         scores_of_ones = weight_score_temp[indices_of_ones]
#         threshold_low = scores_of_ones.quantile(0.33333)
#         threshold_high = scores_of_ones.quantile(0.66666)

#         w_b[indices_of_ones] = torch.where(
#             scores_of_ones < threshold_low,
#             1.0,
#             torch.where(scores_of_ones < threshold_high, 2, 3),
#         )

#         w_b = w_b.reshape(weight_score.shape)

#         ctx.mask = w_b
#         ctx.decay = decay
#         ctx.weight = weight

#         return weight * w_b.T

#     @staticmethod
#     def backward(ctx, grad_output):

#         # (weight_score,) = ctx.saved_tensors
#         return None, grad_output.T - ctx.decay * (3 - ctx.mask) * ctx.weight.T, None, None
#         # return None, grad_output.T, None, None


# class GetNMSubnet(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, weight, weight_score, N, M, decay=0.0015):
#         ctx.save_for_backward(weight_score)

#         length = weight_score.numel()
#         group = int(length / M)

#         weight_score_temp = weight_score.detach().abs().reshape(group, M)
#         index = torch.argsort(weight_score_temp, dim=1)[:, : int(M - N)]

#         w_b = torch.ones(weight_score_temp.shape, device=weight_score_temp.device)
#         w_b = w_b.scatter_(dim=1, index=index, value=0)

#         w_b = w_b.reshape(weight_score.shape)
#         ctx.mask = w_b
#         ctx.decay = decay
#         ctx.weight = weight

#         return weight * w_b.T

#     @staticmethod
#     def backward(ctx, grad_output):

#         (weight_score,) = ctx.saved_tensors
#         return None, grad_output.T - ctx.decay * (1 - ctx.mask) * ctx.weight.T, None, None


# class GetNMSubnet(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, scores, N, M, zeros, ones):
#         # scoresをグループ化し、各グループ内でN個の要素を残すようにプルーニング
#         batch_size, length = scores.shape
#         group_size = M
#         num_groups = (length + group_size - 1) // group_size

#         padding_size = num_groups * group_size - length
#         zeros_padded = zeros[:, :padding_size].expand(batch_size, -1).to(scores.device)
#         scores_padded = torch.cat([scores, zeros_padded], dim=1)
#         scores_abs = scores_padded.abs().view(batch_size, num_groups, group_size)
#         _, indices = scores_abs.topk(k=M - N, dim=2, largest=False)
#         mask = torch.ones_like(scores_abs, dtype=torch.bool).scatter_(2, indices, 0)

#         mask = mask.view(batch_size, -1)[:, :length]
#         out = torch.where(mask, ones.to(scores.device), zeros.to(scores.device))
#         ctx.save_for_backward(mask)

#         num_ones_per_row = out.sum(dim=1)
#         assert torch.all(num_ones_per_row == num_ones_per_row[0]), "各行の1の数が一致していません"

#         return out

#     @staticmethod
#     def backward(ctx, g):
#         return g, None, None, None, None


class SparseParameter(SparseModule):
    def __init__(self, heads, out_channels, args=None, layer=0):
        super().__init__()

        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity

        if args.init_mode_linear is not None:
            self.init_mode = args.init_mode_linear
        else:
            self.init_mode = args.init_mode

        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(1, heads, out_channels))
        self.weight_score = nn.Parameter(torch.ones((1, heads, out_channels)))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        # self.mask=torch.ones(list(self.weight.size())+[2]).cuda()

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False
        self.sparsity_value = args.linear_sparsity

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_mask,
            scale=self.init_scale_score,
            layer=layer,
        )
        self.init_param_(
            self.weight,
            init_mode=self.init_mode,
            scale=self.init_scale,
            sparse_value=self.sparsity_value,
            layer=layer,
        )

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        self.enable_abs_comp = args.enable_abs_comp

    def forward(self, threshold, manual_mask=None):
        weight_score = self.weight_score
        if manual_mask is None:
            if self.enable_abs_comp is False:
                subnet = GetSubnet.apply(
                    weight_score,
                    threshold,
                    self.weight_zeros,
                    self.weight_ones,
                )
                pruned_weight = self.weight * subnet
            else:
                subnet = GetSubnet.apply(
                    weight_score.abs(),
                    threshold,
                    self.weight_zeros,
                    self.weight_ones,
                )
                pruned_weight = self.weight * subnet
        else:
            pruned_weight = self.weight * manual_mask

        return pruned_weight

    def rerandomize(self, mode, la, mu, sparsity, manual_mask=None):
        if sparsity is None:
            sparsity = self.sparsity

        if manual_mask is None:
            if self.enable_abs_comp is False:
                rate = self.rerand_rate
                mask = GetSubnet.apply(
                    torch.sigmoid(self.weight_score),
                    sparsity * rate,
                    self.weight_zeros,
                    self.weight_ones,
                )
            else:
                rate = self.rerand_rate
                mask = GetSubnet.apply(
                    torch.sigmoid(self.weight_score.abs()),
                    sparsity * rate,
                    self.weight_zeros,
                    self.weight_ones,
                )
        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(
            self.weight,
            mask,
            mode,
            la,
            mu,
            self.init_mode,
            scale,
            self.weight_twin,
            param_score=self.weight_score,
        )


class SparseParameterMulti_mask(SparseModule):
    def __init__(self, heads, out_channels, args=None, layer=0):
        super().__init__()

        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity

        if args.init_mode_linear is not None:
            self.init_mode = args.init_mode_linear
        else:
            self.init_mode = args.init_mode

        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(1, heads, out_channels))
        self.weight_score = nn.Parameter(torch.ones((1, heads, out_channels)))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        # self.mask=torch.ones(list(self.weight.size())+[2]).cuda()

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False
        self.sparsity_value = args.linear_sparsity

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_mask,
            scale=self.init_scale_score,
            layer=layer,
        )
        self.init_param_(
            self.weight,
            init_mode=self.init_mode,
            scale=self.init_scale,
            sparse_value=self.sparsity_value,
            layer=layer,
        )

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        self.enable_abs_comp = args.enable_abs_comp
        self.sp_list = args.sparsity_list
        self.en_mask = args.enable_mask
        self.args = args

    def forward(self, threshold, manual_mask=None):
        weight_score = self.weight_score
        if self.en_mask is False:  # this is the default mode with 1 mask
            if manual_mask is None:
                if self.enable_abs_comp is False:
                    subnet = GetSubnet.apply(
                        weight_score,
                        threshold,
                        self.weight_zeros,
                        self.weight_ones,
                    )
                    pruned_weight = self.weight * subnet
                else:
                    subnet = GetSubnet.apply(
                        weight_score.abs(),
                        threshold,
                        self.weight_zeros,
                        self.weight_ones,
                    )
                    pruned_weight = self.weight * subnet
            else:
                pruned_weight = self.weight * manual_mask
        else:  # multimask
            if manual_mask is None:
                subnets = []
                for threshold_v in threshold:
                    if self.enable_abs_comp is False:
                        subnet = GetSubnet.apply(
                            weight_score,
                            threshold_v,
                            self.weight_zeros,
                            self.weight_ones,
                        )
                        subnets.append(subnet)
                    else:
                        subnet = GetSubnet.apply(
                            weight_score.abs(),
                            threshold_v,
                            self.weight_zeros,
                            self.weight_ones,
                        )
                        subnets.append(subnet)
                combined_subnet = torch.stack(subnets).sum(dim=0)
                pruned_weight = self.weight * combined_subnet
            else:
                pruned_weight = self.weight * manual_mask

        # if self.args.flowgnn_debug:
        #     # 現在の時刻を取得
        #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        #     # self.weightをbinとtxtで保存
        #     torch.save(self.weight, f"att_weight_{current_time}.bin")
        #     # np.savetxt(f"att_weight_{current_time}.txt", self.weight.detach().cpu().numpy())

        #     # combined_subnetをbinとtxtで保存
        #     (combined_subnet, f"att_combined_subnet_{current_time}.bin")
        #     # np.savetxt(
        #     #     f"att_combined_subnet_{current_time}.txt", combined_subnet.detach().cpu().numpy()
        #     # )

        #     # pruned_weightをbinとtxtで保存
        #     torch.save(pruned_weight, f"att_pruned_weight_{current_time}.bin")
        #     # np.savetxt(
        #     #     f"att_pruned_weight_{current_time}.txt", pruned_weight.detach().cpu().numpy()
        #     # )

        return pruned_weight

    def rerandomize(self, mode, la, mu, sparsity, manual_mask=None):
        if sparsity is None:
            sparsity = self.sparsity

        if manual_mask is None:
            if self.enable_abs_comp is False:
                rate = self.rerand_rate
                mask = GetSubnet.apply(
                    torch.sigmoid(self.weight_score),
                    sparsity * rate,
                    self.weight_zeros,
                    self.weight_ones,
                )
            else:
                rate = self.rerand_rate
                mask = GetSubnet.apply(
                    torch.sigmoid(self.weight_score.abs()),
                    sparsity * rate,
                    self.weight_zeros,
                    self.weight_ones,
                )
        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(
            self.weight,
            mask,
            mode,
            la,
            mu,
            self.init_mode,
            scale,
            self.weight_twin,
            param_score=self.weight_score,
        )
