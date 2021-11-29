import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from word2vec.w2v_pytorch.model import Word2VecTrainer


EPOCHS = 10
MAX_EARLY_STOP = 3
EMBEDDING_SIZE = 10
LR = 0.001


def train(context_list, dict_word_to_index, device, lr=LR, epochs=EPOCHS, max_early_stop=MAX_EARLY_STOP):
    '''

    Args:
        context_list: List of context and target tuple, i.e.: (['target', ['context', 'words'])
        dict_word_to_index: Dictionary mapping word to index
        lr: Initial learning rate
        epochs: Max epochs to run
        max_early_stop: Early stop if loss hasn't improved by max_early_stop
    '''
    vocab_size = len(dict_word_to_index)
    model = Word2VecTrainer(EMBEDDING_SIZE, vocab_size)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SparseAdam(model.parameters(), lr=lr)

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


def build_context_target_list(text_list, window_size=3):
    '''
    Return pairs of context and targets. e.g.:
    ```
    "The dog eats cats" for window of 2 --> [(['The', 'eats', 'cats'], 'dog'),
                                             (['The','dog', 'cats'], 'eats'),
                                             (['dog', 'eats'], 'cats')]
    ```
    Args:
        text_list: Text split as list
        window_size: Context window size


    '''
    context_list = []
    for i, word in enumerate(text_list):
        context_ = []
        if (i <= window_size):
            context_.extend(text_list[0:i])
        else:
            context_.extend(text_list[i - window_size:i])
        if (i >= (len(text_list) - window_size)):
            context_.extend(text_list[i + 1:])
        else:
            context_.extend(text_list[i + 1: i + window_size + 1])

        context_list.append((word, context_))
    return context_list



