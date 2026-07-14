"""TextCNN 模型：多尺寸一维卷积 + 最大池化做文本分类。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=C.EMBED_DIM,
                 num_filters=C.NUM_FILTERS, filter_sizes=C.FILTER_SIZES,
                 num_classes=C.NUM_CLASSES, dropout=C.DROPOUT,
                 pad_idx=C.PAD_IDX):
        super().__init__()
        # 填充 id 对应的词向量初始化为 0，避免影响训练
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # 每种卷积核尺寸一个 Conv1d
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        # Conv1d 期望 (batch, channels, length)
        x = x.permute(0, 2, 1)
        # 每个卷积输出 (batch, num_filters, out_len) -> 最大池化 -> (batch, num_filters)
        features = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        # 拼接所有尺寸的特征
        out = torch.cat(features, dim=1)
        out = self.dropout(out)
        return self.fc(out)
