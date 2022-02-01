# DeepMatch_Torch

![Python Version](https://img.shields.io/badge/Python-3.5%2B-green)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.1%2B-green)
![PyPI](https://img.shields.io/pypi/v/deepmatch-torch)

PyTorch version of [DeepMatch](https://github.com/shenweichen/DeepMatch).


DeepMatch_Torch is a PyTorch Version deep matching model library for recommendations & advertising. It's easy to **train models** and to **export representation vectors** for user and item which can be used for **ANN search**.You can use any complex model with `model.fit()`and `model.predict()` .

Let's [**Get Started!**](https://deepmatch.readthedocs.io/en/latest/Quick-Start.html) or [**Run examples**](./examples/colab_MovieLen1M_YoutubeDNN.ipynb) !

## Install
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

## Tutorial 

You can run [example](https://github.com/bbruceyuan/DeepMatch-Torch/tree/main/examples) by following steps.

```bash
cd examples
python run_youtubednn.py or run_fm_dssm.py
```

## Models List

|    Model   | Paper| 
| :------: | :----------- |
|  [FM](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/fm.py)  | [ICDM 2010][Factorization Machines](https://www.researchgate.net/publication/220766482_Factorization_Machines) |
| [DSSM](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/dssm.py) | [CIKM 2013][Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)    |
| [YoutubeDNN](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/youtubednn.py)     | [RecSys 2016][Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)            |
| [NCF](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/ncf.py)  | [WWW 2017][Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)       |
| [MIND](https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/mind.py) | [CIKM 2019][Multi-interest network with dynamic routing for recommendation at Tmall](https://arxiv.org/pdf/1904.08030)  |
