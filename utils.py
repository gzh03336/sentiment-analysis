"""训练/推理用的工具函数：设备选择、评估、模型保存。"""
import torch

import config as C


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """在给定数据集上计算准确率。"""
    model.eval()
    correct, total = 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        preds = model(texts).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def save_model(model):
    """保存模型权重到 config 中定义的路径。"""
    import os
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), C.BEST_MODEL_PATH)


def load_model(model, path=C.BEST_MODEL_PATH):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
