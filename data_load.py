import torch
import logging
import random
from config import *
from utils import myutils

import numpy
from torchtext.data import Dataset, TabularDataset, Field, NestedField, BucketIterator, Iterator
from torchtext.vocab import Vectors
from torchtext.datasets import SequenceTaggingDataset

def tokenize(sequence: str): #for .data
    return [sequence]

def post_process(arr, _):
    return [[int(item) for item in arr_item] for arr_item in arr]

TEXT = Field(sequential=True, tokenize=tokenize, include_lengths=True)
CHAR_NESTING = Field(tokenize = list, lower = True)
CHAR = NestedField(CHAR_NESTING)
TAG = Field(sequential=True, tokenize=tokenize, is_target=True, unk_token=None)
#Fields = [('id', None), (('text','char'), (TEXT, CHAR)), ('tag', TAG)] #
#Fields = [(('text','char'), (TEXT, CHAR)), ('tag', TAG)] #for 
Fields = [(('text','char'), (TEXT, CHAR)), ('tag', TAG)] #for allrels .data iob

class MY_DATA():
    def __init__(self):
        self.TAG = TAG
        self.TEXT = TEXT
        self.CHAR = CHAR

    def get_seqdataset(self, path: str, fields= Fields, separator='\t'): #for mscholar separator is space ' '
        dataset = SequenceTaggingDataset(path, fields=fields, separator=separator)
        return dataset

    #for IOB normal data
    def get_splitdata(self, path: str, fields= Fields, separator='\t', split=CONFIG['split']):
        dataset = SequenceTaggingDataset(path, fields=fields, separator=separator)
        #train_dt, test_dt, dev_dt = dataset.split(split_ratio=split, random_state=random.getstate())
        train_dt, test_dt, dev_dt = dataset.split(split_ratio=split, random_state=None)
        logging.info (f'size train {len(train_dt)}, valid {len(dev_dt)}, test {len(test_dt)}')
        return train_dt, dev_dt, test_dt

    def get_vocab(self, *dataset):
        if CONFIG['pretrained']:
            vec = Vectors(CONFIG['pretrained'])
            TEXT.build_vocab(*dataset, vectors=vec)
        else:
            TEXT.build_vocab(*dataset)
        CHAR.build_vocab(*dataset)
        TAG.build_vocab(*dataset)
        return TEXT.vocab, TAG.vocab, CHAR.vocab

    def get_vectors(self, path: str):
        vectors = Vectors.path
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=CONFIG['batch_size'], device=DEVICE, sort_key=lambda x: len(x.text), sort_within_batch=True):
        return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key, sort_within_batch=sort_within_batch)

mydata = MY_DATA()