"""推理入口：加载训练好的模型，对输入文本进行情感预测。"""
import sys

import torch

import config as C
from data.dataset import load_vocab, text_to_ids
from model.textcnn import TextCNN
from utils import get_device, load_model

LABELS = {0: "负面", 1: "正面"}


def predict(text, model, vocab, device):
    ids = torch.tensor([text_to_ids(text, vocab)], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(ids), dim=1)[0]
    pred = prob.argmax().item()
    return LABELS[pred], prob[pred].item()


def main():
    device = get_device()
    vocab = load_vocab(C.VOCAB_PATH)
    model = TextCNN(vocab_size=len(vocab)).to(device)
    load_model(model)
    print("模型加载完成，输入文本进行情感分析（输入 q 退出）。\n")

    while True:
        text = input("请输入影评文本: ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue
        label, conf = predict(text, model, vocab, device)
        print(f"  -> 预测: {label} (置信度 {conf:.2%})\n")


if __name__ == "__main__":
    main()
