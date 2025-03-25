---
title: 'BingoGCN: Towards Scalable and Efficient GNN Acceleration with Fine-Grained Partitioning and SLT'

---

# BingoGCN: Towards Scalable and Efficient GNN Acceleration with Fine-Grained Partitioning and SLT

 
Jiale Yan, Hiroaki Ito, Yuta Nagahara, Kazushi Kawamura, Masato Motomura, Thiem Van Chu, Daichi Fujiki.

---

## Overview of evalutions

We propose a GCN algorithm and accelerator co-design framework called BingoGCN.

* ***On the algorithm level***, BingoGCN integrates Strong Lottery Tickets (SLT) during the combination stage and introduces Cross-Partition Message Quantization (CMQ) in the aggregation stage to reduce computational overhead. Additionally, graph partitioning algorithms are employed to evaluate the impact of graph structure on performance.

* ***On the hardware level***, BingoGCN features a dedicated accelerator that exploits the sparsity and structure of the graphs produced by the algorithm-level optimizations, thereby further enhancing acceleration efficiency.

 
## Usage of the Provided Codes

This release contains codes focusing on the algorithmic part described in the paper.

> **Offline METIS Graph Partitioning for Fig.1 in the paper.**  
> Dataset: OGBN-Arxiv with varying numbers of partitions
> Code location: ./offline-METIS/
 
> Environment Setup

```bash
# Python and PyTorch versions
Python version: 3.10.8
PyTorch version: 1.13.0+cu117

# Install required packages
pip install pymetis         # For METIS partitioning
pip install torch_geometric # For accessing datasets
```

> **SLT Training codes for GNNs.**  
> Dataset: XXXX
> Code location: XXXXX
> With crossponding figures in the paper: Fig.19
 
 
> Environment Setup

```bash
# Python and PyTorch versions
Python version: XXXX
PyTorch version: XXXX

# Install required packages
XXXX
```
 
> **CMQ codes for GNNs.**  
> Dataset: XXXX
> Code location: XXXXX
> With crossponding figures in the paper: Fig.15
 
 
> Environment Setup

```bash
# Python and PyTorch versions
Python version: XXXX
PyTorch version: XXXX

# Install required packages
XXXX
```

Supported models
- 3/4 layer GCNs with 192 hidden dimensions. 
 
Supported datasets
How to change the dataset "XXXX change path = load_dir"
- Cora
- CiteSeer
- Pubmed
- OGBN-Arxiv
- OGBN-Reddit
- Other graph-level tasks.
  
   