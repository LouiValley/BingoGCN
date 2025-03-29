# Artifact of the BingoGCN Paper, ISCA 2025

Jiale Yan, Hiroaki Ito, Yuta Nagahara, Kazushi Kawamura, Masato Motomura, Thiem Van Chu, Daichi Fujiki.

---

This repository provides models and programs required for the artifact evaluation of the "BingoGCN: Towards Scalable and Efficient GNN Acceleration with Fine-Grained Partitioning and SLT" paper published in ISCA 2025. 

The artifact of this paper includes models and programs to reproduce the key contributions of fine-grained partitioning and SLT algorithms.

## Directory Structure
- [BingoGCN](/BingoGCN/): Main project directory containing the core code and scripts.
- [Expected_results](/Expected_results/): Directory storing expected results for artifact evaluation.
- [\_\_outputs\_\_](/__outputs__/): Directory storing checkpoints of the baseline models.
- [env](/env/): Directory for environment-related files.
- [scripts](/scripts/): Directory containing scripts for job execution used in the project.

## Requirements
- NVIDIA GPUs with more than 32 GB of memory. The light version, using only small datasets, can run on a GPU with less memory.
- OS supporting CUDA 11.6. The code is tested on Ubuntu 20.04.
- Anaconda, PyTorch 1.13, CUDA 11.6.
- A minimum of 4 GB of free disk space.

## Environmental Setup
Install CUDA 11.6 on a machine running a supported OS. Install Anaconda or Miniconda. Then, download the repository and install dependencies as follows. 

```bash
git clone https://github.com/LouiValley/BingoGCN.git
cd BingoGCN

conda env create -f env/conda.yml
conda activate BingoGCN
pip install -r env/requirements.txt
```

## Run Artifact
Use one of the scripts in the [scripts](/scripts/) directory. 

### To reproduce all data points:
```bash
sh scripts/jobs_all.sh
```
> [!NOTE]
> This will take approximately two days with 8 GPUs and can take weeks on a single GPU. For the artifact evaluation, we recommend the other options below. 

### To reproduce "Ours" only:
```bash
sh scripts/jobs_ours.sh
```
This script evaluates the data points of our proposed design, reducing the single-GPU runtime to a few days. 

### To reproduce small datasets in "Ours":
```bash
sh scripts/jobs_ours_light.sh
```
This script reduces repetition counts and skips evaluation on large datasets. This will only take 2~3 hours to complete. 


## Post-Experiment Steps
Each experiment makes a directory under ./logs. 
After the experiments, run:

```bash
python BingoGCN/log_to_csv.py
```
This script aggregates the results into CSV files within each log directory.

Finally, compare these results with the ones in the expected_results directory.
You can verify the data points in the figures with them. There can be minor discrepancies in the numbers due to non-deterministic factors such as RNG states.

## Experiment Customizations
See [Customize.md](/Customize.md).

## Citation
If you use *BingoGCN*, please cite this paper:

> Jiale Yan, Hiroaki Ito, Yuta Nagahara, Kazushi Kawamura, Masato Motomura, Thiem Van Chu, and Daichi Fujiki,
> *"BingoGCN: Towards Scalable and Efficient GNN Acceleration with Fine-Grained Partitioning and SLT,"*
> In Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA'25)

```
@inproceedings{bingogcn,
  title={BingoGCN: Towards Scalable and Efficient GNN Acceleration with Fine-Grained Partitioning and SLT},
  author={Yan, Jiale and Ito, Hiroaki and Nagahara, Yuta and Kawamura, Kazushi and Motomura, Masato and Chu, Thiem Van and Fujiki, Daichi},
  booktitle={Proceedings of the 52nd Annual International Symposium on Computer Architecture}, 
  year={2025}
}
```

## Licensing

This repository is available under a [MIT license](/LICENSE).
