import torch

d_model = 64
vocab_size = 65
block_size = 32
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
lr = 1e-3
max_iters = 5000
eval_interval = 100
