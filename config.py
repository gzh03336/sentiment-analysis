"""集中管理训练超参数与路径配置，方便统一调整。"""
import os

# ---- 路径 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# ---- 数据 ----
MAX_VOCAB_SIZE = 30000   # 词表最大长度
MAX_SEQ_LEN = 256        # 句子统一长度（截断/补齐）
BATCH_SIZE = 64

# ---- 模型 ----
EMBED_DIM = 128          # 词向量维度
NUM_FILTERS = 100        # 每种尺寸的卷积核数量
FILTER_SIZES = [3, 4, 5] # 卷积核尺寸
DROPOUT = 0.5
NUM_CLASSES = 2          # 正面 / 负面

# ---- 训练 ----
EPOCHS = 5
LEARNING_RATE = 1e-3
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# 预留词表索引
PAD_IDX = 0
UNK_IDX = 1
