import random
import torch


def get_data():
    """
    Quickstart:

    words, stoi, itos = get_data()
    """
    words = open("../../data/names.txt", "r").read().splitlines()

    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return words, stoi, itos


def build_dataset(words: list[str], stoi: object, block_size=3):
    X, Y = [], []

    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


def load_split_data(words: list[str], stoi: object, block_size=3):
    """
    Quickstart:

    Xtr, Ytr, Xdev, Ydev, Xte, Yte = load_split_data()
    """
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)
    return Xtr, Ytr, Xdev, Ydev, Xte, Yte
