import argparse
import torch
from word2vec.w2v_pytorch.train import train, build_context_target_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train word2vec on selected text.')
    parser.add_argument('--text', type=str, help='Text to train on if no file is passed')
    # TODO: Allow file to be passed
    # TODO: Allow device to be passed
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=3)
    args = parser.parse_args()
    text_list = args.text.split()
    dict_word_to_index = {word: i for i, word in enumerate(set(text_list))}
    context_list = build_context_target_list(text_list, args.window_size)
    model = train(context_list, dict_word_to_index, epochs=args.epochs, device=device)
    torch.save(model.state_dict(), './model.pt')


if __name__ == '__main__':
    main()