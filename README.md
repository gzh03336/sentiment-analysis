# 影评情感分析 (TextCNN)

基于 PyTorch 的文本情感分析项目，使用 **TextCNN** 模型对 IMDB 影评进行正面/负面二分类。

## 项目结构

```
sentiment-analysis/
├── config.py            # 超参数与路径配置
├── data/
│   └── dataset.py       # 数据加载、分词、词表构建
├── model/
│   └── textcnn.py       # TextCNN 模型
├── utils.py             # 评估、模型保存等工具函数
├── train.py             # 训练入口
├── predict.py           # 推理入口（交互式）
├── requirements.txt     # 依赖
└── .gitignore
```

## 环境依赖

- Python 3.9+
- PyTorch 2.0+

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

首次运行会自动下载 IMDB 数据集到 `data_cache/`，并在 `checkpoints/` 下保存最佳模型与词表。

```bash
python train.py
```

> 国内用户如遇 HuggingFace 数据集下载失败，可设置镜像：
> ```bash
> # Linux/macOS
> export HF_ENDPOINT=https://hf-mirror.com
> # Windows PowerShell
> $env:HF_ENDPOINT="https://hf-mirror.com"
> ```

训练 5 个 epoch 后，在测试集上一般可达 **86%+** 准确率。

### 2. 交互式推理

训练完成后，可对任意输入文本进行情感预测：

```bash
python predict.py
```

示例：

```
请输入影评文本: This movie is absolutely wonderful, the acting is great!
  -> 预测: 正面 (置信度 99.87%)

请输入影评文本: Boring and predictable. Total waste of time.
  -> 预测: 负面 (置信度 98.42%)
```

## 模型说明

**TextCNN** 由 Kim (2014) 提出，核心思想是用多种尺寸的一维卷积核捕捉不同长度的 n-gram 特征，再通过最大池化提取最重要的特征用于分类。

本项目使用的结构：

| 组件 | 配置 |
|------|------|
| Embedding | 维度 128 |
| 卷积核尺寸 | 3, 4, 5（各 100 个） |
| Dropout | 0.5 |
| 输出 | 二分类 |

所有超参数集中在 `config.py`，可自行调整。

## 可扩展方向

- [ ] 加入 LSTM / BiLSTM 模型对比
- [ ] 使用预训练词向量 (GloVe / Word2Vec)
- [ ] 替换为中文数据集 (如 ChnSentiCorp)
- [ ] 接入 HuggingFace Transformers 做 BERT 微调

## 数据集

[IMDB Movie Reviews](https://huggingface.co/datasets/imdb)：5 万条电影评论，标注正面/负面，训练集与测试集各 2.5 万条。

## 参考

- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP.
