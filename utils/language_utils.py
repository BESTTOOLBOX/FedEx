"""Utils for language models."""


import json
from collections import Counter
import os
import re
import ssl
import urllib.request
import logging
import numpy as np
import os.path as osp


logger = logging.getLogger(__name__)

def save_local_data(dir_path,
                    train_data=None,
                    train_targets=None,
                    test_data=None,
                    test_targets=None,
                    val_data=None,
                    val_targets=None):
    r"""
    Save data to disk. Source: \
    https://github.com/omarfoq/FedEM/blob/main/data/femnist/generate_data.py
    Args:
        train_data: x of train data
        train_targets: y of train data
        test_data: x of test data
        test_targets: y of test data
        val_data: x of validation data
        val_targets:y of validation data
    Note:
        save ``(`train_data`, `train_targets`)`` in ``{dir_path}/train.pt``, \
        ``(`val_data`, `val_targets`)`` in ``{dir_path}/val.pt`` \
        and ``(`test_data`, `test_targets`)`` in ``{dir_path}/test.pt``
    """
    import torch
    if (train_data is not None) and (train_targets is not None):
        torch.save((train_data, train_targets), osp.join(dir_path, "train.pt"))

    if (test_data is not None) and (test_targets is not None):
        torch.save((test_data, test_targets), osp.join(dir_path, "test.pt"))

    if (val_data is not None) and (val_targets is not None):
        torch.save((val_data, val_targets), osp.join(dir_path, "val.pt"))


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric
    Args:
        url (string): The url of target file.
        folder (string): The target folder.
    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        logger.info(f'File {file} exists, use existing file.')
        return path

    logger.info(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path


# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[" \
              "]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    '''returns a list of character indices
    Arguments:
        word: string
    :returns:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words
    Arguments:
        line: string representing phrase to be split
    :returns:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab
    Arguments:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values
    :returns:
        integer list
    '''
    bag = [0] * len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def target_to_binary(label):
    return int(label == 1)


def token_to_ids(texts, vocab):
    to_ret = [[vocab[word] for word in line] for line in texts]
    return np.array(to_ret)


def label_to_index(labels):
    counter = Counter(labels)
    sorted_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    label_list = [x[0] for x in sorted_tuples]
    return [label_list.index(x) for x in labels]