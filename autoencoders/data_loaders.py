import sys
sys.path.append("../")

import os
import numpy as np
import h5py
from pathlib import Path
import torch
from torch.utils import data
from collections import defaultdict
from progress.bar import Bar
import argparse
from tokenizers import (CharBPETokenizer, SentencePieceBPETokenizer)

TOKENIZER_LIST = ["CharBPETokenizer",
                  "SentencePieceBPETokenizer"]


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_text_file", type=str,
                        help="The text file to load dataset from.")
    parser.add_argument("output_file", type=str,
                        help="The .h5 file to output the dataset to.")
    parser.add_argument("batch_size", type=int,
                        help="The batch size which the dataset will batched into.")

    # From-Scratch Data Loading
    parser.add_argument("-v", "--vocab_file", default="vocab", type=str,
                        help="The file to output the dataset vocab / tokenizer model to.")
    parser.add_argument("-mf", "--min_freq", type=int, default=5,
                        help="The min frequency to accept a word in vocab.")
    parser.add_argument("-mw", "--max_words", type=int, default=30000,
                        help="The max number of words to have in the vocab.")

    # Pre-Trained Tokenizer
    parser.add_argument("-t", "--tokenizer", required=True, type=str,
                        help="Specify the tokenizer to use.", choices=TOKENIZER_LIST)
    parser.add_argument("--location", type=str,
                        help="Path where to find the tokenizer", default=None)

    params, _ = parser.parse_known_args()
    return params


def generate_dataset_with_tokenizer(TEXT_FILE,
                                    DATASET_FILE,
                                    TOKENIZER,
                                    MAX_SENTENCE_LENGTH,
                                    BATCH_SIZE=64,
                                    MIN_FREQ=5,
                                    MAX_FILE_SIZE_BATCHES=2000000,
                                    MAX_WORDS=30000):  # note: with a batch size of 64 and MAX_FILE_SIZE_BATCHES 200k each file equates to roughly 1.5-2GB):

    TOKENIZER.train([TEXT_FILE], vocab_size=MAX_WORDS, special_tokens=[
                    "[PAD]", "<unk>", "<SOS>", "<EOS>"], min_frequency=MIN_FREQ)
    TOKENIZER.save("/".join(DATASET_FILE.split("/")[:-1]), "tokenizer")

    ###### Save sequences to dataset #####
    file_counter = 0
    dataset = h5py.File(DATASET_FILE + str(file_counter) + ".h5", 'w')
    sent_counter = 0
    batch_counter = 0
    ided_sentences_by_length = defaultdict(list)
    with Bar('Writing sentences to hdf5') as bar:
        with open(TEXT_FILE, 'r') as f:

            def save_to_h5(sentlist, length):
                nonlocal dataset, batch_counter, file_counter, MAX_FILE_SIZE_BATCHES

                lengths_batch = np.array(
                    [length] * len(sentlist), dtype=np.uint32)
                sentences_batch = np.zeros(
                    (len(sentlist), length), dtype=np.uint32)
                for i, s in enumerate(sentlist):
                    for j, index in enumerate(s):
                        sentences_batch[i, j] = index

                g = dataset.create_group("BATCH" + str(batch_counter))
                g.create_dataset('data', data=sentences_batch)
                g.create_dataset('length', data=lengths_batch)

                batch_counter += 1

                if (batch_counter % MAX_FILE_SIZE_BATCHES) == 0:
                    dataset.close()
                    file_counter += 1
                    dataset = h5py.File(
                        DATASET_FILE + str(file_counter) + ".h5", 'w')

            for line in f:

                ided = TOKENIZER.encode(
                    "<SOS>" + line.rstrip() + "<EOS>").ids

                ided_len = len(ided)
                if ided_len >= 2 and ided_len <= MAX_SENTENCE_LENGTH:
                    ided_sentences_by_length[ided_len].append(ided)
                    # ided_sentences.append(sentence_ids)
                    sent_counter += 1

                    n_sent_by_len = len(ided_sentences_by_length[ided_len])
                    if n_sent_by_len == BATCH_SIZE:
                        save_to_h5(
                            ided_sentences_by_length[ided_len], ided_len)
                        ided_sentences_by_length[ided_len] = []

                    bar.next()

            # push out all remaining sentences
            for k, v in ided_sentences_by_length.items():
                if len(v) > 0:
                    save_to_h5(v, k)

    dataset.close()


def _tokens_to_index(token_list, word2index):
    index_list = [word2index["<SOS>"]]
    for t in token_list:
        if t in word2index:
            index_list.append(word2index[t])
    index_list.append(word2index["<EOS>"])
    return index_list


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info_type = {}
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index).astype("int64")
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get length
        y = self.get_data("length", index).astype("int64")
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds[()], file_path)

                    # find the beginning index of the hdf5 file we are looking
                    # for
                    file_idx = next(i for i, v in enumerate(
                        self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded
                    # it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'],
                               'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        if type not in self.data_info_type:
            data_info_type = [
                di for di in self.data_info if di['type'] == type]
            self.data_info_type[type] = data_info_type
        else:
            data_info_type = self.data_info_type[type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


def get_tokenizer(tokenizer, location='bert-base-uncased'):
    if tokenizer == "BERT":
        return BertTokenizer.from_pretrained(location)
    else:
        if location is not None:
            return eval(tokenizer)(vocab_file=location + '-vocab.json',
                                   merges_file=location + '-merges.txt')
        else:
            return eval(tokenizer)()


if __name__ == "__main__":
    params = get_params()
    os.makedirs(os.path.dirname(params.output_file), exist_ok=True)
    if params.tokenizer:
        generate_dataset_with_tokenizer(params.input_text_file,
                                        params.output_file,
                                        get_tokenizer(
                                            params.tokenizer, location=params.location),
                                        100,
                                        BATCH_SIZE=params.batch_size,
                                        MIN_FREQ=params.min_freq,
                                        MAX_WORDS=params.max_words)
