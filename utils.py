from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
import logging
import numpy
import torch
from config import *
import itertools
import os

class MY_UTILS():
    def set_logger(self, log_path):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)

    def count_elapsed_time(self, start, end):
        elapsed_time = end - start
        e_min = int(elapsed_time)/60
        e_sec = int(elapsed_time - (elapsed_time*60))
        return e_min, e_sec

    def save_result(self, result, mid, path=None,):
        if not os.path.isdir(path):
            os.mkdir(path)
        result_path = os.path.join(path, 'pner_'+mid+'_result.txt')
        fo=open(result_path, 'w')
        for sen in result:
            w,t,p=sen[0][0],sen[0][1],sen[0][2]
            try:
                assert len(w) == len(t) == len(p), 'not equal rnn'
            except:
                w=sen[0][0][1:]
                assert len(w) == len(t) == len(p), 'not equal bert'
            for w1,t1,p1 in zip(w, t, p):
                print (w1,t1,p1,sep='\t',file=fo)
            print ('',file=fo)
        fo.close()
        logging.info(f'Result saved to {result_path}')
    
    #for bert ff
    def argmax_accuracy(self, pred_y, true_y, tag_pad_idx):
        max_pred_y = pred_y.argmax(dim = 2, keepdim = True) # get the index of the max probability
        pred_y = max_pred_y.squeeze(2)
        pred_y = pred_y.cpu().detach().numpy()
        pred_y = numpy.transpose(pred_y).tolist()
        true_y = true_y.cpu().detach().numpy()
        true_y = numpy.transpose(true_y).tolist()
        batch_p, batch_r, batch_f1 = [],[],[]
        for t,p in zip(true_y, pred_y):
            assert len(t) == len(p), 'not equal up!'
            pad_elements = [s for s, c in enumerate(t) if c == tag_pad_idx]
            t = [x for i,x in enumerate(t) if i not in pad_elements]
            p = [y for j,y in enumerate(p) if j not in pad_elements]
            assert len(t) == len(p), 'not equal!'
            p_s = precision_score(t,p, average='macro')
            r_s = recall_score(t,p, average='macro')
            f1_s = f1_score (t,p, average='macro')
            batch_p.append(p_s)
            batch_r.append(r_s)
            batch_f1.append(f1_s)
        avg_p = sum(batch_p) / len(batch_p)
        avg_r = sum(batch_r) / len(batch_r)
        avg_f1 = sum(batch_f1) / len(batch_f1)
        return avg_p, avg_r, avg_f1

    def argmax_accuracy_save(self, pred_y, true_y, tag_pad_idx, btokenizer, tag_vocab, sent):
        max_pred_y = pred_y.argmax(dim = 2, keepdim = True) # get the index of the max probability
        pred_y = max_pred_y.squeeze(2)
        pred_y = pred_y.cpu().detach().numpy()
        pred_y = numpy.transpose(pred_y).tolist()
        true_y = true_y.cpu().detach().numpy()
        true_y = numpy.transpose(true_y).tolist()
        sent = sent.cpu().detach().numpy()
        sent = numpy.transpose(sent).tolist()
        pair = []
        batch_p, batch_r, batch_f1 = [],[],[]
        for t,p,s in zip(true_y, pred_y, sent):
            assert len(t) == len(p) == len(s), 'not equal up!'
            pad_elements = [s for s, c in enumerate(t) if c == tag_pad_idx]
            t = [x for i,x in enumerate(t) if i not in pad_elements]
            p = [y for j,y in enumerate(p) if j not in pad_elements]
            assert len(t) == len(p), 'not equal!'
            p_s = precision_score(t,p, average='macro')
            r_s = recall_score(t,p, average='macro')
            f1_s = f1_score (t,p, average='macro')
            batch_p.append(p_s)
            batch_r.append(r_s)
            batch_f1.append(f1_s)
            true = [tag_vocab.itos[i] for i in t]
            pred = [tag_vocab.itos[i] for i in p]
            sen = [btokenizer.convert_ids_to_tokens(i) for i in s]
            pair.append([sen, true, pred])
        avg_p = sum(batch_p) / len(batch_p)
        avg_r = sum(batch_r) / len(batch_r)
        avg_f1 = sum(batch_f1) / len(batch_f1)
        return avg_p, avg_r, avg_f1, pair

    def per_class(self, classk, y_true, y_pred):
        try:
            return classification_report(y_true, y_pred, output_dict=True)[classk].get('f1-score')
        except:
            return None

    def get_score_iterator(self, tag_pred, tag_true, tag_pad_idx):
        tag_true = tag_true.cpu().detach().numpy()
        tag_true = numpy.transpose(tag_true).tolist()
        assert len(tag_true) == len(tag_pred), 'len batch must be equal!'
        batch_p, batch_r, batch_f1 = [],[],[]
        for t,p in zip(tag_true, tag_pred):
            pad_elements = [s for s, c in enumerate(t) if c == tag_pad_idx]
            t = [x for i,x in enumerate(t) if i not in pad_elements]
            #p = [y for j,y in enumerate(p) if j not in pad_elements]
            assert len(t) == len(p), 'not equal even after remove pad!'
            p_s = precision_score(t,p, average='macro')
            r_s = recall_score(t,p, average='macro')
            f1_s = f1_score (t,p, average='macro')
            batch_p.append(p_s)
            batch_r.append(r_s)
            batch_f1.append(f1_s)
        avg_p = sum(batch_p) / len(batch_p)
        avg_r = sum(batch_r) / len(batch_r)
        avg_f1 = sum(batch_f1) / len(batch_f1)

        return avg_p, avg_r, avg_f1

    def get_score_iterator_save(self, tag_pred, tag_true, tag_pad_idx, w_vocab, tag_vocab, sent):
        tag_true = tag_true.cpu().detach().numpy()
        tag_true = numpy.transpose(tag_true).tolist()
        assert len(tag_true) == len(tag_pred), 'len batch must be equal!'
        batch_p, batch_r, batch_f1 = [],[],[]
        sent = sent.cpu().detach().numpy()
        sent = numpy.transpose(sent).tolist()
        pair = []
        for t,p,s in zip(tag_true, tag_pred, sent):
            pad_elements = [s for s, c in enumerate(t) if c == tag_pad_idx]
            t = [x for i,x in enumerate(t) if i not in pad_elements]
            #p = [y for j,y in enumerate(p) if j not in pad_elements]
            assert len(t) == len(p), 'not equal even after remove pad!'
            p_s = precision_score(t,p, average='macro')
            r_s = recall_score(t,p, average='macro')
            f1_s = f1_score (t,p, average='macro')
            batch_p.append(p_s)
            batch_r.append(r_s)
            batch_f1.append(f1_s)
            true = [tag_vocab.itos[i] for i in t]
            pred = [tag_vocab.itos[i] for i in p]
            sen = [w_vocab.itos[i] for i in s]
            pair.append([sen, true, pred])
        avg_p = sum(batch_p) / len(batch_p)
        avg_r = sum(batch_r) / len(batch_r)
        avg_f1 = sum(batch_f1) / len(batch_f1)
        return avg_p, avg_r, avg_f1, pair

    #when the data is not yet iterator
    def get_score(self, model, sent, char, tag_true, w_vocab, c_vocab, tag_vocab):
        maxl = len(max(sent, key=len))
        pad = 1
        vector_char = [[c_vocab.stoi[c] for c in w] for w in sent]
        for c in vector_char:
            c += [pad] * (maxl - len(c))
        vector_char = torch.unsqueeze(torch.tensor(vector_char),0).to(DEVICE)
        vector_text = torch.tensor([w_vocab.stoi[x] for x in sent]).view(-1,1).to(DEVICE)
        vector_lentext = torch.tensor([len(vector_text)]).to(DEVICE)
        vector_prediction = model(vector_text, vector_lentext, vector_char)[0]
        #tag_pred = [tag_vocab.itos[i] for i in vector_prediction]
        tag_pred = vector_prediction
        tag_true = [tag_vocab.stoi[i] for i in tag_true]
        assert len(tag_true) == len(tag_pred), 'Length of gold and prediction must be equal!'
        f1_s = f1_score (tag_true, tag_pred, average='macro')
        p_s = precision_score(tag_true, tag_pred, average='macro')
        r_s = recall_score(tag_true, tag_pred, average='macro')
        pair = [sent, tag_true, tag_pred]
        return p_s, r_s, f1_s, pair

myutils = MY_UTILS()