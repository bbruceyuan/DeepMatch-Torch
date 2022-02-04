# DeepMatch-Torch

![Python Version](https://img.shields.io/badge/Python-3.5%2B-green)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.1%2B-green)
![PyTorch-Lightning](https://img.shields.io/badge/PyTorch--Lightning-1.5%2B-green)
![PyPI](https://img.shields.io/pypi/v/deepmatch-torch)


[DeepMatch-Torch](https://github.com/bbruceyuan/DeepMatch-Torch) is a `PyTorch` version of [DeepMatch](https://github.com/shenweichen/DeepMatch).


[DeepMatch-Torch](https://github.com/bbruceyuan/DeepMatch-Torch) is a PyTorch Version deep matching model library for recommendations & advertising. It's easy to **train models** and to **export representation vectors** for user and item which can be used for **ANN search**. You can use any complex model with `model.fit()` and `model.predict()`. And you can keep same habit of using DeepMatch.


Let's [**Get Started!**](https://deepmatch.readthedocs.io/en/latest/Quick-Start.html) or [**Run examples**](./examples/colab_MovieLen1M_YoutubeDNN.ipynb) !


## 1. Install
1. Install `deepmatch-torch` by `pip`
```bash
pip install deepmatch-torch
```

2. Install `deepmatch-torch` through source code
```bash
git clone https://github.com/bbruceyuan/DeepMatch-Torch.git
cd DeepMatch-Torch
python setup.py install
```

## 2. Tutorial 

You can run [example](https://github.com/bbruceyuan/DeepMatch-Torch/tree/main/examples) by following steps.

```bash
cd examples
python run_youtubednn.py or run_fm_dssm.py
```


## 3. Models List

|    Model   | Paper| 
| :------: | :----------- |
|  [FM](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/fm.py)  | [ICDM 2010][Factorization Machines](https://www.researchgate.net/publication/220766482_Factorization_Machines) |
| [DSSM](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/dssm.py) | [CIKM 2013][Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)    |
| [YoutubeDNN](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/youtubednn.py)     | [RecSys 2016][Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)            |
| [NCF](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/ncf.py)  | [WWW 2017][Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)       |
| [MIND](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/mind.py) | [CIKM 2019][Multi-interest network with dynamic routing for recommendation at Tmall](https://arxiv.org/pdf/1904.08030)  |


## TODO
- [ ] simplify model config. now only support kwargs, but config is a elegant choice.
- [ ] fix `MIND` only support CPU train bug.


## Acknowledgments
Especially thanks to [DeepMatch](https://github.com/shenweichen/DeepMatch). This project relies highly on DeepMatch. Additionally, I used the `PLBaseModel` design from [torchTS](https://github.com/Rose-STL-Lab/torchTS).

Thanks to this awesome projects.

## 中文 README

[DeepMatch-Torch](https://github.com/bbruceyuan/DeepMatch-Torch) 是一个 `PyTorch` 版本的 [DeepMatch](https://github.com/shenweichen/DeepMatch)。


[DeepMatch-Torch](https://github.com/bbruceyuan/DeepMatch-Torch) 是一个用于广告推荐的召回模型库，可以非常简单地训练模型和导致 `user` 和 `item` 的 **vertor 表示**，你可以用这个 user/item 的表示进行 ANN 近似检索。在 DeepMatch-Torch 中，你可以保持和 DeepMatch 一样的习惯，通过 `model.fit()` 和 `model.predict()` 进行模型的训练和预测。

更多的中文文档细节可以参见：[中文文档](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/README_ZH.md)

## 1. 安装
- 步骤一: 安装 PyTorch, 按照[官网指引](https://pytorch.org/get-started/locally/)
- 步骤二：
    - 通过 pip 安装：`pip install deepmatch-torch` 
    - 通过源码安装
```bash
git clone https://github.com/bbruceyuan/DeepMatch-Torch.git
cd DeepMatch-Torch
python setup.py install
```

## 2. 快速上手
进入 [example](https://github.com/bbruceyuan/DeepMatch-Torch/tree/main/examples) 目录查看相关代码，使用方式和 DeepMatch 几乎保持一致。

```bash
cd examples
python run_youtubednn.py or run_fm_dssm.py
```

