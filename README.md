# Grug: A Unified Gradient Regularization Method for Enhancing Stability and Robustness in Heterogeneous Graph Neural Networks

This is the official repo for the paper Unifying Gradient Regularization for Heterogeneous Graph Neural Networks.
**TL;DR:**: *Grug* regularizes gradients from both propagated messages and node features during message-passing and offers a unified framework that integrates graph topology and node features.

## Requirements
+ torch>=1.12.1
+ torch-geometric>=2.2.0
+ dgl>=0.9.1

## Experiments

### Datasets
+ [ACM]([http://localhost/](https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)
+ [DBLP](http://localhost/](https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)
+ [IMDB](http://localhost/](https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)
+ [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)
+ [amazon](http://localhost/](https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)
+ [LastFM](http://localhost/](https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset)

### Baseline methods
+ [RGCN](https://gitcode.com/tkipf/relational-gcn?utm_source=csdn_github_accelerator)
+ [RGAT](https://github.com/shenwzh3/RGAT-ABSA)
+ [HGT](https://github.com/acbull/pyHGT)
+ [HAN](https://github.com/Jhy1993/HAN)
+ [SimpleHGNN](https://github.com/THUDM/HGB?tab=readme-ov-file)

### Backbone models
+ [DropEdge](https://github.com/DropEdge/DropEdge)
+ [DropNode](Graph random neural networks for semi-supervised learning on graphs)
+ [DropMessage](https://github.com/zjunet/DropMessage)
+ [FLAG](https://github.com/devnkong/FLAG)

## Quick Start
You can directly run the source code of *Grug*.

---
```
python main.py -m model -t task -d dataset -a alpha -b beta -M M 
```

Optional arguments are as follows:
+ `-m model` name of models.
+ `-t task` name of task.
+ `-d dataset` name of datasets.
+ `-a alpha` parameter of uniform distribution on original message matrix.
+ `-b beta` parameter of uniform distribution on original node matrix.
+ `-M M` number of iterations.

e.g.:
```
python main.py -m RGCN -t node_classification -d ACM -a 0.01 -b 0.01 -M 3
```

