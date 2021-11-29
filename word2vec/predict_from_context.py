import argparse
import pickle
import torch
from w2v_pytorch.train import train
from w2v_pytorch.predict import predict
from w2v_pytorch.utils import build_context_target_list, clean_and_split_text


def main():

    parser = argparse.ArgumentParser(description='Train word2vec on selected text.')
    parser.add_argument('--text', type=str, help='Text to train on if no file is passed', default=None)
    # TODO: Allow device to be passed
    parser.add_argument('--window_size', type=int, default=3)



    args = parser.parse_args()
    if args.file:
        file = open(args.file, "r")
        text = file.read()
    else:
        text = args.text
    text_list = clean_and_split_text(text)

    dict_word_to_index = pickle.load(open("../word_to_index.pkl", "rb"))
    result = predict(context_size=args.window_size,
                     context=text_list,
                     path='./model.pt',
                     dict_word_to_index=dict_word_to_index)
    print('Predicted word: ', result)


if __name__ == '__main__':
    main()
