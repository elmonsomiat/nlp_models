
EPOCHS = 10
MAX_EARLY_STOP = 3
EMBEDDING_SIZE = 10
LR = 0.001


def build_context_target_list(text_list, window_size=3):
    '''
    Return pairs of context and targets. Ignore the ones which would be outside of the window e.g.:
    ```
    "The dog eats cats" for window of 1 --> [(['The', 'eats'], 'dog'),
                                             (['dog', 'cats'], 'eats')]
    ```
    Args:
        text_list: Text split as list
        window_size: Context window size


    '''
    context_list = []
    for i, word in enumerate(text_list):
        context_ = []
        if (i <= window_size):
            continue
        else:
            context_.extend(text_list[i - window_size:i])
        if (i >= (len(text_list) - window_size)):
            continue
        else:
            context_.extend(text_list[i + 1: i + window_size + 1])
        context_list.append((word, context_))
    return context_list

