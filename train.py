import torch
import logging
import sys
import os

from tqdm import tqdm
from config import *

import torch.nn as nn

from data_load import mydata
from utils import myutils

from model import RNN_CRF, MY_CONFIG, CHAR_ENCODER_CNN

class MY_SEQ_TAG():
    def __init__(self):
        self._model = None
        self._word_vocab = None
        self._tag_vocab = None
        self._char_vocab = None

    def train(self, trainp, devp=None, testp=None, full_split=False, full_mode=False, **kwargs):
        logging.info (f'Loading dataset...')

        if full_split:
            tr_dt, dev_dt, te_dt = mydata.get_splitdata(trainp)
            logging.info (f'load {trainp}, train dev test {len(tr_dt)}, {len(dev_dt)}, {len(te_dt)}')
        else:
            tr_dt = mydata.get_seqdataset(trainp)
            logging.info (f'load {trainp}, size {len(tr_dt)}')
            dev_dt = mydata.get_seqdataset(devp)
            logging.info (f'load {devp}, size {len(dev_dt)}')
            te_dt = mydata.get_seqdataset(testp)
            logging.info (f'load {testp}, size {len(te_dt)}')

        word_vocab, tag_vocab, char_vocab = mydata.get_vocab(tr_dt) 

        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        self._char_vocab = char_vocab
        self._pad_token_idx = word_vocab.stoi[mydata.TEXT.pad_token]
        self._pad_tag_idx = tag_vocab.stoi[mydata.TAG.pad_token]
        self._pad_char_idx = char_vocab.stoi[mydata.CHAR.pad_token]

        train_it = mydata.get_iterator(tr_dt, batch_size=CONFIG['batch_size'])
        dev_it = mydata.get_iterator(dev_dt, batch_size=1)

        logging.info ('#Preparing the model...')
        config = MY_CONFIG(word_vocab, tag_vocab, char_vocab, self._pad_token_idx, self._pad_tag_idx, self._pad_char_idx, **kwargs)
        char_encoder = CHAR_ENCODER_CNN(config.char_vocab_size, config.char_emb_dim, config.char_cnn_filter_size, config.word_embedding_dim, config.pad_char_idx)
        logging.info (f'CONFIG: {CONFIG}')
        model = RNN_CRF(char_encoder, config).to(DEVICE)
        self._model = model
        optimizer = torch.optim.Adam(model.parameters())
        best_valid_acc = 0.0

        for ep in range(config.epoch):
            logging.info (f'#Training epoch {ep} ...')
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            for batch in tqdm(train_it):
                model.zero_grad()
                loss =  model.loss(batch.text[0], batch.text[1], batch.char, batch.tag)
                loss = (-loss)
                predictions = model(batch.text[0], batch.text[1], batch.char)
                s_p,s_r,s_f1 = myutils.get_score_iterator(predictions, batch.tag, self._pad_tag_idx)
                epoch_loss += loss.view(-1).cpu().data.tolist()[0]
                epoch_acc += s_f1
                loss.backward()
                optimizer.step()
            logging.info (f'Train: Loss {epoch_loss/len(train_it): .4f}')
            logging.info (f'Train: Acc {(epoch_acc/len(train_it)*100): .2f}')
            if devp or full_mode:
                logging.info (f'#Evaluating epoch {ep} ...')
                p,r,f1 = self._validate_iterator(dev_it)
                logging.info (f'Dev: P,R,F1 {p: .3f} {r: .3f} {f1: .3f}')

            if f1 > best_valid_acc:        
                best_valid_acc = f1
                model.save(path=config.save_path, mid=config.mid)
                logging.info (f'Best Model Acc: {best_valid_acc*100: .2f}')
            logging.info ('\n')
        config.save(path=config.save_path, mid=config.mid)

        if full_mode:
            logging.info (f'#Testing ...')
            te_it = mydata.get_iterator_test(te_dt, batch_size=1)
            config_path = os.path.join(config.save_path, 'pner_'+config.mid+'_config.pkl')
            model_path = os.path.join(config.save_path, 'pner_'+config.mid+'_model.pkl')
            self.load_model(config_path, model_path)
            #p,r,f1,a,b,c,d,e = self._validate_iterator(te_it)
            p,r,f1, result = self._validate_iterator(te_it, s=True)
            logging.info (f'Test: P,R,F1 {p: .3f} {r: .3f} {f1: .3f}')
            myutils.save_result(result, mid=config.mid, path=config.save_path)

    def _validate_iterator(self, dev_it, s=False):
        self._model.eval()
        f1_sent, p_sent, r_sent, result  = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dev_it):
                predictions = self._model(batch.text[0], batch.text[1], batch.char)
                if s:
                    s_p,s_r,s_f1,pair = myutils.get_score_iterator_save(predictions, batch.tag, self._pad_tag_idx, self._word_vocab, self._tag_vocab, batch.text[0])
                    result.append(pair)
                else:
                    s_p,s_r,s_f1 = myutils.get_score_iterator(predictions, batch.tag, self._pad_tag_idx)
                p_sent.append(s_p)
                r_sent.append(s_r)
                f1_sent.append(s_f1)
        avg_p = sum(p_sent) / len(p_sent)
        avg_r = sum(r_sent) / len(r_sent)
        avg_f1 = sum(f1_sent) / len(f1_sent)
        if s:
            return avg_p, avg_r, avg_f1, result 
        else:
            return avg_p, avg_r, avg_f1

    def load_model(self, config_path, model_path):
        config = MY_CONFIG.load(self, config_path)
        char_encoder = CHAR_ENCODER_CNN(config.char_vocab_size, config.char_emb_dim, config.char_cnn_filter_size, config.word_embedding_dim, config.pad_char_idx).to(DEVICE)
        model = RNN_CRF(char_encoder, config).to(DEVICE)
        model.load(model_path)
        self._model = model
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
        self._char_vocab = config.char_vocab

myseq = MY_SEQ_TAG()