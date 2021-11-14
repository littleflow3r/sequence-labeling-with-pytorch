# process-ner

<b>(1) Prepare the dataset (text, tag), tab separated. The tag column uses IOB format.</b>

example: please see data/sample.data

<b>(2) Edit the configuration in config.py</b>

P.S. If using pretrained embedding (e.g. word2vec), set the word_embedding option to the dimension of the pretrained embedding. If using BERT model, change the LR to be very-very small.

<b>(3) Run the main.py </b>

    3.1) Full mode (training, validation, testing) on automatically random splitted data. Set the split ratio for train, dev, and test data on "split" in config.py. Additionally, change the "seed" on config.py to different values on different training (e.g. for cross-validation)
    
    e.g. python3 main.py full --id trainID --model rnn_crf/bert_ff --data_path data/all.data --epoch 15
    
    3.2. Full mode (training, validation, testing) on carefully splitted data. Split the data into train, dev, and test manually.
    
    e.g. python3 main.py main --id trainID --model rnn_crf/bert_ff --train_path data/train.data --dev_path data/dev.data --test_path data/test.data --epoch 15
    
    3.2) Prediction/inference on input sentence

    e.g. python3 main.py predict --s "YOUR SENTENCE HERE." --config_path saves/pner_trainID_config.pkl --model_path saves/pner_trainID_model.pkl --model rnn_crf/bert_ff




Contact: susanti.yuni@fujitsu.com
