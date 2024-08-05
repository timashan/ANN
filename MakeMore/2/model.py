import torch


class NGram:
    def __init__(self, vocab_size=27, n_embed=2, block_size=3, n_hidden=100) -> None:
        self.C = torch.randn((vocab_size, n_embed))

        self.W1_fan_in = n_embed * block_size
        self.W1 = (
            torch.randn((self.W1_fan_in, n_hidden)) * (5 / 3) / (self.W1_fan_in**0.5)
        )  # 0.2
        # self.b1 = torch.randn(n_hidden) * 0.01

        self.W2 = torch.randn((n_hidden, vocab_size)) * 0.01
        self.b2 = torch.randn(vocab_size) * 0

        self.bngain = torch.ones((1, n_hidden))
        self.bnbias = torch.zeros((1, n_hidden))
        # buffers
        self.bnmean_running = torch.zeros((1, n_hidden))
        self.bnstd_running = torch.ones((1, n_hidden))

        self.parameters = [
            self.C,
            self.W1,
            # self.b1,
            self.W2,
            self.b2,
            self.bngain,
            self.bnbias,
        ]
        for p in self.parameters:
            p.requires_grad = True

        print(f"no.of paramters: {sum([p.nelement() for p in self.parameters])}")

    def forward(self, X: torch.Tensor):
        emb = self.C[X]
        emb = emb.view(-1, self.W1_fan_in)
        h = emb @ self.W1  # + self.b1
        bnmeani = h.mean(0, keepdim=True)
        bnstdi = h.std(0, keepdim=True)
        h = self.bngain * (h - bnmeani) / bnstdi + self.bnbias

        with torch.no_grad():
            # momentum: 0.001
            self.bnmean_running = self.bnmean_running * 0.999 + bnmeani * 0.001
            self.bnstd_running = self.bnstd_running * 0.999 + bnstdi * 0.001

        h = torch.tanh(h)
        logits = h @ self.W2 + self.b2
        return logits

    def predict(self, X):
        emb = self.C[X]
        emb = emb.view(-1, self.W1_fan_in)
        h = emb @ self.W1  # + self.b1
        h = self.bngain * (h - self.bnmean_running) / self.bnstd_running + self.bnbias
        h = torch.tanh(h)
        logits = h @ self.W2 + self.b2
        return logits
