import torch
import torch.nn as nn
import torch.optim as optim

from word2vec.w2v_pytorch.model import Word2VecTrainer
from word2vec.w2v_pytorch.utils import EPOCHS, LR, MAX_EARLY_STOP, EMBEDDING_SIZE




def train(context_list,
          dict_word_to_index,
          context_size,
          device,
          lr=LR,
          epochs=EPOCHS,
          max_early_stop=MAX_EARLY_STOP,
          embedding_size=EMBEDDING_SIZE):
    '''

    Args:
        context_list: List of context and target tuple, i.e.: (['target', ['context', 'words'])
        dict_word_to_index: Dictionary mapping word to index
        lr: Initial learning rate
        epochs: Max epochs to run
        max_early_stop: Early stop if loss hasn't improved by max_early_stop
    '''
    vocab_size = len(dict_word_to_index)
    model = Word2VecTrainer(context_size, embedding_size, vocab_size)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stop_count = 0
    for step in range(epochs):
        losses = []
        for target, context in context_list:
            context_tensor = torch.tensor([dict_word_to_index[w] for w in context],
                                          dtype=torch.long,
                                          device=device)
            model.zero_grad()
            pred = model(context_tensor)
            loss = loss_func(pred, torch.tensor([dict_word_to_index[target]],
                                                dtype=torch.long,
                                                device=device))
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
            if len(losses) > 1:
                if losses[0] >= losses[-1]:
                    stop_count += 1
            if stop_count >= max_early_stop:
                print('Early stopping...')
                return model
    return model



