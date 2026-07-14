"""训练入口：加载数据 -> 训练 TextCNN -> 保存最佳模型。"""
import torch
import torch.nn as nn
from tqdm import tqdm

import config as C
from data.dataset import get_dataloaders, save_vocab
from model.textcnn import TextCNN
from utils import get_device, evaluate, save_model


def main():
    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader, vocab = get_dataloaders()
    print(f"词表大小: {len(vocab)}")

    # 保存词表，供推理使用
    import os
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    save_vocab(vocab, C.VOCAB_PATH)

    model = TextCNN(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{C.EPOCHS}")
        for texts, labels in pbar:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} 完成 | 平均损失: {total_loss / len(train_loader):.4f} | 测试准确率: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_model(model)
            print(f"  -> 新的最佳模型已保存 (acc={acc:.4f})")

    print(f"\n训练结束，最佳测试准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
