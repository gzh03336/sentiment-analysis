"""数据加载与预处理：IMDB 影评二分类。"""
import json
import os
import re
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import config as C

# 去除 HTML 标签的简单正则
_HTML_RE = re.compile(r"<[^>]+>")


def tokenize(text: str):
    """简单分词：转小写 -> 去 HTML 标签 -> 按非字母数字切分。"""
    text = _HTML_RE.sub("", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def build_vocab(texts, max_size=C.MAX_VOCAB_SIZE):
    """根据训练集文本统计词频，构建 {词: id} 词表。前两位留给 pad/unk。"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {C.PAD_TOKEN: C.PAD_IDX, C.UNK_TOKEN: C.UNK_IDX}
    for word, _ in counter.most_common(max_size - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def text_to_ids(text, vocab):
    """单条文本转 id 序列，并截断/补齐到 MAX_SEQ_LEN。"""
    ids = [vocab.get(tok, C.UNK_IDX) for tok in tokenize(text)]
    ids = ids[: C.MAX_SEQ_LEN]
    ids += [C.PAD_IDX] * (C.MAX_SEQ_LEN - len(ids))
    return ids


def collate_fn(batch, vocab):
    """DataLoader 批处理：把原始文本转成张量。batch 为 dict 列表。"""
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    text_tensor = torch.tensor(
        [text_to_ids(t, vocab) for t in texts], dtype=torch.long
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return text_tensor, label_tensor


def get_dataloaders(vocab=None):
    """加载 IMDB 数据集并返回 train/test DataLoader。

    若不传入 vocab，则从训练集自动构建并保存。
    """
    ds = load_dataset("imdb", cache_dir=C.DATA_CACHE_DIR)

    if vocab is None:
        vocab = build_vocab(ds["train"]["text"])
        save_vocab(vocab, C.VOCAB_PATH)

    from functools import partial
    fn = partial(collate_fn, vocab=vocab)

    train_loader = DataLoader(
        ds["train"], batch_size=C.BATCH_SIZE, shuffle=True, collate_fn=fn
    )
    test_loader = DataLoader(
        ds["test"], batch_size=C.BATCH_SIZE, shuffle=False, collate_fn=fn
    )
    return train_loader, test_loader, vocab
