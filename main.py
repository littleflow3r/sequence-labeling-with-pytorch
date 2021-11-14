import os
import time
import sys
import argparse
import logging

from config import *

from utils import myutils

from train import myseq

def main():
    cmd = argparse.ArgumentParser('Training!')
    cmd.add_argument('--id', required=True, default="train1", help='id of the training (string)')
    cmd.add_argument("--data_path", required=True, help="the path to the full data")
    cmd.add_argument("--model", required=True, default="rnn_crf", help="model: rnn_crf, bert_ff, or bert_crf")
    cmd.add_argument("--epoch", type=int, default=10, help='epochs')
    args = cmd.parse_args(sys.argv[2:])
    
    myutils.set_logger(os.path.join(CONFIG['save_path'], 'pner_'+str(args.id)+'_train.log'))
    logging.info (f'#Training full mode (include validation+testing)')
    logging.info (f'#Training with {args.model}...')
    myseq.train(args.data_path, epoch=args.epoch, mid=str(args.id), full_split=True, full_mode=True)

if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'main':
            main()
        else:
            print ('wrong args syntax!')
    else:
        print('Usage: {0} [main] [test] [predict] [options]'.format(sys.argv[0]), file=sys.stderr)

    end = time.time()
    e_min, e_sec = myutils.count_elapsed_time(start, end)
    logging.info (f'#Total elapsed time: {e_min}m {e_sec}s')