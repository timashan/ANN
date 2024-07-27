import torch


class BiGram:
    def __init__(self, n_vocab=27) -> None:
        self.n_vocab = n_vocab

    def train(self, words: list[str]):
        N = torch.zeros(self.n_vocab, self.n_vocab, dtype=torch.int32)
        chars = sorted(set("".join(words)))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi["."] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

        for w in words:
            chs = ["."] + list(w) + ["."]
            for c1, c2 in zip(chs, chs[1:]):
                ix1 = self.stoi[c1]
                ix2 = self.stoi[c2]
                N[ix1, ix2] += 1

        # +1 is model smoothing, similar to regularization
        self.P = (N + 1) / (N + 1).sum(1, keepdim=True)

        log_likelihood = 0
        n = 0
        for w in words:
            for c1, c2 in zip(w, w[1:]):
                x1 = self.stoi[c1]
                x2 = self.stoi[c2]
                prob = self.P[x1, x2]
                logprob = torch.log(prob)
                log_likelihood += logprob
                n += 1
        nnl = -log_likelihood
        return nnl / n

    def generate(self, n=1):
        out = []

        for _ in range(n):
            word = []
            ix = 0
            while True:
                p = self.P[ix]
                ix = torch.multinomial(p, num_samples=1, replacement=True).item()
                word.append(self.itos[ix])
                if ix == 0:
                    break
            out.append("".join(word))
        return out
