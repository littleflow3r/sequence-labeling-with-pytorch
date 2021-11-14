# sequence-labeling-with-pytorch

<b>(1) Prepare the dataset (text, tag), tab separated. The tag column uses IOB format.</b>

example: please see data/sample.train

<b>(2) Edit the configuration in config.py</b>

P.S. If using pretrained embedding (e.g. word2vec), set the word_embedding option to the dimension of the pretrained embedding.

<b>(3) Run the main.py </b>
Full mode (training, validation, testing) on automatically random splitted data. Set the split ratio for train, dev, and test data on "split" in config.py. Additionally, change the "seed" on config.py to different values on different training (e.g. for cross-validation)
    
e.g. python3 main.py main --id trainID --model rnn_crf --data_path data/all.data --epoch 30

The model, config, and prediction result will be saved in 'saves' dir.
