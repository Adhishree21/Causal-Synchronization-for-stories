import re
import torch

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.is_fitted = vocab is not None

    def fit(self, texts):
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.is_fitted = True

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def __call__(self, text):
        words = self._tokenize(text)
        return torch.tensor([self.vocab.get(word, self.vocab["<UNK>"]) for word in words], dtype=torch.long)

    def size(self):
        return len(self.vocab)
