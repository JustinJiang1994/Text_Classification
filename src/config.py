# -*- coding: utf-8 -*-



class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                # "vocab_path": "../data/cnews/cnews.vocab.txt",
                "vocab_path": "../data/rumor/cnews.vocab.txt",
                # "trainingSet_path": "../data/cnews/cnews.train.txt",
                "trainingSet_path": "../data/rumor/train_list.txt",
                # "valSet_path": "../data/cnews/cnews.val.txt",
                "valSet_path": "../data/rumor/val_list.txt",
                # "testingSet_path": "../data/cnews/cnews.test.txt",
                "testingSet_path": "../data/rumor/test_list.txt"
            },
            "CNN_training_rule": {
                "embedding_dim": 64,
                "seq_length": 200,
                "num_classes": 2,

                "conv1_num_filters": 128,
                "conv1_kernel_size": 1,

                "conv2_num_filters": 128,
                "conv2_kernel_size": 1,

                "vocab_size": 5000,

                "hidden_dim": 256,

                "dropout_keep_prob": 0.5,
                "learning_rate": 1e-3,

                "batch_size": 64,
                "epochs": 5,

                "print_per_batch": 50,
                "save_per_batch": 500
            },
            "LSTM": {
                "seq_length": 300,
                "num_classes": 2,
                "vocab_size": 5000,
                "batch_size": 64
            },
            "result": {
                "CNN_model_path": "CNN_model.h5",
                "LSTM_model_path": "LSTM_model.h5"
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]
