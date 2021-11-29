import argparse
import pickle
import torch
from w2v_pytorch.train import train
from w2v_pytorch.predict import predict
from w2v_pytorch.utils import build_context_target_list, clean_and_split_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train word2vec on selected text.')
    parser.add_argument('--text', type=str, help='Text to train on if no file is passed', default=None)
    parser.add_argument('--file', type=str, help='File with text', default=None)
    # TODO: Allow device to be passed
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=3)
    args = parser.parse_args()
    if args.file:
        file = open(args.file, "r")
        text = file.read()
    else:
        text = args.text
    text_list = clean_and_split_text(text)

    dict_word_to_index = {word: i for i, word in enumerate(set(text_list))}
    outfile = open("../word_to_index.pkl", "wb")
    pickle.dump(dict_word_to_index, outfile)
    context_list = build_context_target_list(text_list, args.window_size)

    model = train(context_size=args.window_size,
                  context_list=context_list,
                  dict_word_to_index=dict_word_to_index,
                  epochs=args.epochs,
                  device=device)
    torch.save(model.state_dict(), './model.pt')


if __name__ == '__main__':
    main()
