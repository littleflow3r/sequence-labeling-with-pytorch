import torch
import numpy
import random
torch.cuda.empty_cache()

CONFIG = {
    'gpu': True,
    'lr': 5e-5,
    'batch_size': 16,
    'dropout': 0.5,
    'seed': 2021,
    
    'word_embedding_dim': 200,
    'rnn_hidden_dim': 200,
    'rnn_num_layers': 2,
    'rnn_bidirectional': True,

    'char_emb': True,
    'char_emb_dim': 100,
    'char_cnn_filter_size': 3,
    
    'split': [0.8, 0.05, 0.15],
    'save_path': './saves',

    'pretrained': False
    #'pretrained': '../../../pretrained/glove.6B.200d.txt'

}

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = CONFIG['seed']
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() and CONFIG['gpu'] else 'cpu')