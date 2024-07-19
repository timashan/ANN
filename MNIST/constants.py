import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
n_iters = 1_000
lr = 0.1
n_hidden = 100
