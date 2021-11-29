import torch.nn.functional as F
import torch.nn as nn


class Word2VecTrainer(nn.Module):

    def __init__(self, context_size, emb_size, vocab_size):
        super(Word2VecTrainer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(2 * context_size * emb_size, vocab_size)

    def forward(self, context_words):
        emb = self.embeddings(context_words).view((1, -1))
        hidden = self.linear(emb)
        return F.log_softmax(hidden, dim=1)
