import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 0.005
current_loss = 0
plot_steps, print_steps = 1_000, 5_000
n_iters = 100_000
hidden_size = 128
