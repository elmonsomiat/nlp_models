import torch
from word2vec.w2v_pytorch.model import Word2VecTrainer
from word2vec.w2v_pytorch.utils import EMBEDDING_SIZE


def predict(context,
            path,
            dict_word_to_index,
            context_size,
            embedding_size=EMBEDDING_SIZE):
    with torch.no_grad():
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
