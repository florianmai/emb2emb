import numpy as np
import torch


def word_drop(tokenizer, x, p):  # drop words with probability p
    x_ = []
    lengths_ = []
    batch_size = x.size(0)
    special_token_ids = get_special_token_ids(tokenizer)
    for i in range(batch_size):
        words = x[i, :].tolist()
        keep = np.random.rand(len(words)) > p

        # do not drop any of the special symbols
        for i in range(len(words)):
            if words[i] in special_token_ids:
                keep[i] = True

        sent = [w for j, w in enumerate(words) if keep[j]]
        lengths_.append(len(sent))
        sent += [tokenizer.token_to_id("[PAD]")] * (len(words) - len(sent))
        x_.append(sent)

    new_x = torch.LongTensor(x_).contiguous().to(x.device)
    new_l = torch.LongTensor(lengths_).contiguous().to(x.device)
    _, sorted_indices = new_l.sort(dim=0, descending=True)

    # remember original sorting
    original_indices = torch.zeros(len(new_l))
    for i in range(len(new_l)):
        original_indices[sorted_indices[i]] = i
    original_indices = original_indices.long()

    return new_x[sorted_indices, :], new_l[sorted_indices], original_indices


def get_special_token_ids(tokenizer):
    special_tokens = ["[PAD]", "<unk>", "<SOS>", "<EOS>"]
    return [tokenizer.token_to_id(st) for st in special_tokens]


def get_vocab_size(tokenizer):
    return tokenizer._tokenizer.get_vocab_size()


def noisy(tokenizer, x, drop_prob):
    if drop_prob > 0:
        x, lens, indices = word_drop(tokenizer, x, drop_prob)
    return x, lens, indices
