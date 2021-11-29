import torch.nn.functional as F
import torch.nn as nn


class Word2VecTrainer(nn.Module):

    def __init__(self, context_size, emb_size, vocab_size):
        super(Word2VecTrainer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.linear_1 = nn.Linear(2 * context_size * emb_size, 128)
        self.linear_2 = nn.Linear(128, vocab_size)

    def forward(self, context_words):
        emb = self.embeddings(context_words).view((1, -1))
        hidden = F.relu(self.linear_1(emb))
        out = self.linear_2(hidden)
        return F.log_softmax(out, dim=1)
