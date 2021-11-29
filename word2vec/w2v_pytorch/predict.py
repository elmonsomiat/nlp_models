import torch
from w2v_pytorch.model import Word2VecTrainer
from w2v_pytorch.utils import EMBEDDING_SIZE
import torch.nn as nn


def predict(context,
            path,
            dict_word_to_index,
            context_size,
            embedding_size=EMBEDDING_SIZE,
            ):
    with torch.no_grad():
        print(path)
        vocab_size = len(dict_word_to_index)
        model = Word2VecTrainer(context_size=context_size,
                                emb_size=embedding_size,
                                vocab_size=vocab_size)

        model.load_state_dict(torch.load(path))
        model.eval()

        # Generate prediction
        context_tensor = torch.tensor([dict_word_to_index[w] for w in context],
                                      dtype=torch.long)
        prediction = model(context_tensor)
        position = torch.argmax(prediction)
        index_to_word = {dict_word_to_index[i]: i for i in dict_word_to_index}
        return index_to_word[int(position.detach().numpy())]


def predict_closest(word,
                path,
                dict_word_to_index,
                context_size,
                embedding_size=EMBEDDING_SIZE,
                ):
        with torch.no_grad():
            vocab_size = len(dict_word_to_index)
            model = Word2VecTrainer(context_size=context_size,
                                    emb_size=embedding_size,
                                    vocab_size=vocab_size)

            model.load_state_dict(torch.load(path))
            model.eval()
            emb = model.embeddings
            pdist = nn.PairwiseDistance()
            idx = dict_word_to_index[word]
            lookup_tensor_i = torch.tensor([idx], dtype=torch.long)
            v_i = emb(lookup_tensor_i)

            index_to_word = {dict_word_to_index[i]: i for i in dict_word_to_index}
            word_distance = []
            for j in range(len(dict_word_to_index)):
                if j != idx:
                    lookup_tensor_j = torch.tensor([j], dtype=torch.long)
                    v_j = emb(lookup_tensor_j)
                    word_distance.append((index_to_word[j], float(pdist(v_i, v_j))))
            word_distance.sort(key=lambda x: x[1])
            return word_distance[:5]

