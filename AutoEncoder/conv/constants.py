import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
n_iters = 10_000
