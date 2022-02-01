
## DeepMatch-Torch

[DeepMatch-Torch](https://github.com/bbruceyuan/DeepMatch-Torch) 是一个 `PyTorch` 版本的 [DeepMatch](https://github.com/shenweichen/DeepMatch)。这是一个用于广告推荐的召回模型库，可以非常简单地训练模型和导致 `user` 和 `item` 的 **vertor 表示**，你可以用这个 user/item 的表示进行 ANN 近似检索。在 DeepMatch-Torch 中，你可以保持和 DeepMatch 一样的习惯，通过 `model.fit()` 和 `model.predict()` 进行模型的训练和预测。

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

