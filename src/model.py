# -*- coding: utf-8 -*-

"""
Created on 2020-07-19 00:12
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import tensorflow.keras as keras
from config import Config
from preprocess import preprocesser


class TextCNN(object):

    def __init__(self):
        self.config = Config()
        self.pre = preprocesser()

    def model(self):
        num_classes = self.config.get("CNN_training_rule", "num_classes")
        vocab_size = self.config.get("CNN_training_rule", "vocab_size")
        seq_length = self.config.get("CNN_training_rule", "seq_length")

        conv1_num_filters = self.config.get("CNN_training_rule", "conv1_num_filters")
        conv1_kernel_size = self.config.get("CNN_training_rule", "conv1_kernel_size")

        conv2_num_filters = self.config.get("CNN_training_rule", "conv2_num_filters")
        conv2_kernel_size = self.config.get("CNN_training_rule", "conv2_kernel_size")

        hidden_dim = self.config.get("CNN_training_rule", "hidden_dim")
        dropout_keep_prob = self.config.get("CNN_training_rule", "dropout_keep_prob")

        model_input = keras.layers.Input((seq_length,), dtype='float64')
        embedding_layer = keras.layers.Embedding(vocab_size, 256, input_length=seq_length)
        embedded = embedding_layer(model_input)

        # conv1形状[batch_size, seq_length, conv1_num_filters]
        conv_1 = keras.layers.Conv1D(conv1_num_filters, conv1_kernel_size, padding="SAME")(embedded)
        conv_2 = keras.layers.Conv1D(conv2_num_filters, conv2_kernel_size, padding="SAME")(conv_1)
        max_poolinged = keras.layers.GlobalMaxPool1D()(conv_2)

        full_connect = keras.layers.Dense(hidden_dim)(max_poolinged)
        droped = keras.layers.Dropout(dropout_keep_prob)(full_connect)
        relued = keras.layers.ReLU()(droped)
        model_output = keras.layers.Dense(num_classes, activation="softmax")(relued)
        model = keras.models.Model(inputs=model_input, outputs=model_output)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        print(model.summary())
        return model

    def train(self, epochs):
        trainingSet_path = self.config.get("data_path", "trainingSet_path")
        valSet_path = self.config.get("data_path", "valSet_path")
        seq_length = self.config.get("CNN_training_rule", "seq_length")

        x_train, y_train = self.pre.word2idx(trainingSet_path, max_length=seq_length)
        x_val, y_val = self.pre.word2idx(valSet_path, max_length=seq_length)

        model = self.model()
        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=32,
                      epochs=1,
                      validation_data=(x_val, y_val))
if __name__ == '__main__':
    test = TextCNN()
    print(test.model())
