import os
import pickle
import logging

import torch
import torch
import torch.nn as nn
from torchcrf import CRF
from torchtext.vocab import Vectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import torchnlp.nn as nlp
from config import *

class MY_CONFIG():
    def __init__(self, word_vocab, tag_vocab, char_vocab, pad_token_idx, pad_tag_idx, pad_char_idx,  **kwargs):
        super().__init__()
        for name, value in CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self.tag_num = len(self.tag_vocab)
        self.word_vocab_size = len(self.word_vocab)
        self.char_vocab_size = len(self.char_vocab)
        self.pad_token_idx = pad_token_idx
        self.pad_tag_idx = pad_tag_idx
        self.pad_char_idx = pad_char_idx
        for name, value in kwargs.items():
            setattr(self, name, value)

    def save(self, path=None, mid=''):
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, 'pner_'+mid+'_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f'Config saved to {config_path}')

    def load(self, config_path):
        config = None
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logging.info(f'Loading config from {config_path}...')
        return config

class CHAR_ENCODER_CNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, char_cnn_filter_size, hidden_dim, pad_char_idx):
        super().__init__()
        assert char_cnn_filter_size % 2 == 1, "Kernel size must be odd!"
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx = pad_char_idx)
        self.cnn = nn.Conv1d(in_channels =char_emb_dim, out_channels = hidden_dim, kernel_size = char_cnn_filter_size, padding = (char_cnn_filter_size - 1)//2)
    
    def forward(self, chars):
        bsize = chars.shape[0]
        sent_len = chars.shape[1]
        word_len = chars.shape[2]
        embedded = self.embedding(chars) #bsize, sent len, char emb dim, word len
        embedded = embedded.view(-1, word_len, self.char_emb_dim)
        embedded = embedded.permute(0,2,1) #bsize*sent len, char emb dim, word len
        embedded = self.cnn(embedded)
        embedded = embedded.view(bsize, sent_len, self.hidden_dim, word_len) #[batch size, sent len, hid dim, word len]
        embedded = torch.max(embedded, dim = -1).values #bsize, sent len, hid dim
        embedded = embedded.permute(1,0,2) #sent len, bsize, hid dim
        return embedded

class RNN_CRF(nn.Module):
    def __init__(self, char_encoder, args):
        super().__init__()
        self.char_encoder = char_encoder
        self.args = args
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = args.rnn_bidirectional
        self.rnn_num_layers = args.rnn_num_layers
        self.pad_token_idx = args.pad_token_idx
        self.dropout = args.dropout 
        self.save_path = args.save_path
        self.char_emb = args.char_emb

        vocabulary_size = args.word_vocab_size
        embedding_dim = args.word_embedding_dim

        if args.pretrained:
            self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim).from_pretrained(args.word_vocab.vectors)
        else:
            self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        if self.char_emb:
            self.lstm = nn.LSTM(embedding_dim*2, self.rnn_hidden_dim, bidirectional=self.bidirectional,
                            num_layers=self.rnn_num_layers, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(embedding_dim, self.rnn_hidden_dim, bidirectional=self.bidirectional,
                            num_layers=self.rnn_num_layers, dropout=self.dropout)

        # self.lstm = nn.LSTM(embedding_dim*2, self.rnn_hidden_dim, bidirectional=self.bidirectional,
        #                     num_layers=self.rnn_num_layers, dropout=self.dropout)
        self.ff = nn.Linear(self.rnn_hidden_dim, self.tag_num)
        self.crflayer = CRF(self.tag_num)

    def loss(self, x, sent_lengths, char, y):
        mask = torch.ne(x, self.pad_token_idx)
        emissions = self.lstm_forward(x, sent_lengths, char)
        return self.crflayer(emissions, y, mask=mask)

    def forward(self, x, sent_lengths, char):
        mask = torch.ne(x, self.pad_token_idx)
        emissions = self.lstm_forward(x, sent_lengths, char)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, sent_lengths, char):
        word = self.word_embedding(sentence)
        if self.char_emb:
            char = self.char_encoder(char)
            wch = torch.cat((word, char), dim=2)
        else:
            wch = word
        wch = pack_padded_sequence(wch, sent_lengths)
        #wch = pack_padded_sequence(wch, sent_lengths, enforce_sorted=False)
        lstm_out, self.hidden = self.lstm(wch)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(DEVICE)), 'Lengths must be equal!'
        out = lstm_out[:, :, :self.rnn_hidden_dim] + lstm_out[:, :, self.rnn_hidden_dim:]
        y = self.ff(out)
        return y

    def save(self, path=None, mid=''):
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, 'pner_'+mid+'_model.pkl')
        torch.save(self.state_dict(), model_path)
        logging.info(f'Model saved to {model_path}')

    def load(self, model_path):
        map_location = None if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logging.info(f'Loading model from {model_path}...')