
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def word_index_mapping(vocab):
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab):
        word2index[word] = index + 1  # plus 1, since the 0th index is padding
        index2word[index + 1] = word
    return word2index, index2word
