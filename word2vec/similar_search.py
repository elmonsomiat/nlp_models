import argparse
import pickle
from w2v_pytorch.predict import predict_closest


def main():
    parser = argparse.ArgumentParser(description='Train word2vec on selected text.')
    parser.add_argument('--text', type=str, help='Text to train on if no file is passed', default=None)
    # TODO: Allow device to be passed
    parser.add_argument('--window_size', type=int, default=3)

    args = parser.parse_args()
    word = args.text


    dict_word_to_index = pickle.load(open("../word_to_index.pkl", "rb"))
    result = predict_closest(context_size=args.window_size,
                     word=word,
                     path='./model.pt',
                     dict_word_to_index=dict_word_to_index)
    print('Similar words: ', result)


if __name__ == '__main__':
    main()
