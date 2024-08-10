import torch


class Sequential:
    def __init__(self, layers) -> None:
        self.layers = layers

        with torch.no_grad():
            self.layers[-1].weight *= 0.1

        self.parameters = [p for layer in self.layers for p in layer.parameters()]
        # print(sum(p.nelement() for p in self.parameters))
        for p in self.parameters:
            p.requires_grad = True

    def __call__(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for layer in self.layers[:]:
            x = layer(x)
        self.out = x
        return self.out


class Flatten:
    def __init__(self, n=None) -> None:
        self.n = n

    def __call__(self, X: torch.FloatTensor) -> torch.FloatTensor:
        B, T, C = X.shape
        if not self.n:
            self.out = X.view(B, -1)
            return self.out

        self.out = X.view(B, T // self.n, C * self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)

        return self.out

    def parameters(self):
        return []

    def __repr__(self) -> str:
        return "Flatten ()"


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, iX: torch.IntTensor):
        self.out = self.weight[iX]
        return self.out

    def parameters(self):
        return [self.weight]

    def __repr__(self) -> str:
        num_embeddings, embedding_dim = self.weight.shape
        return f"Embedding ({num_embeddings}, {embedding_dim})"


class Linear:

    def __init__(self, fan_in: int, fan_out: int, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.FloatTensor):
        self.out = x @ self.weight
        if self.bias:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])

    def __repr__(self) -> str:
        fan_in, fan_out = self.weight.shape
        return f"Linear ({fan_in}, {fan_out})"


class BatchNorm1d:

    def __init__(self, dim: int, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.FloatTensor):
        # calc forward pass
        if self.training:
            dim = (0, 1) if x.ndim == 3 else 0
            xmean = x.mean(dim, keepdim=True)  # batch mean
            xvar = x.var(dim, keepdim=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return f"BatchNorm1d ({len(self.gamma)})"


class Tanh:
    def __call__(self, x: torch.FloatTensor):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

    def __repr__(self) -> str:
        return "Tanh ()"
