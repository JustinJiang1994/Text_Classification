# -*- coding: utf-8 -*-


import tensorflow.keras as keras
from config import Config
from preprocess import preprocesser
import os
from sklearn import metrics
import numpy as np


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
        embedding_layer = keras.layers.Embedding(vocab_size+1, 256, input_length=seq_length)
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
        model_save_path = self.config.get("result", "CNN_model_path")
        batch_size = self.config.get("CNN_training_rule", "batch_size")

        x_train, y_train = self.pre.word2idx(trainingSet_path, max_length=seq_length)
        x_val, y_val = self.pre.word2idx(valSet_path, max_length=seq_length)

        model = self.model()
        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=1,
                      validation_data=(x_val, y_val))
            model.save(model_save_path, overwrite=True)

    def test(self):
        model_save_path = self.config.get("result", "CNN_model_path")
        testingSet_path = self.config.get("data_path", "testingSet_path")
        seq_length = self.config.get("CNN_training_rule", "seq_length")


        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()

        x_test, y_test = self.pre.word2idx(testingSet_path, max_length=seq_length)
        # print(x_test.shape)
        # print(type(x_test))
        # print(y_test.shape)
        # print(type(y_test))
        pre_test = model.predict(x_test)
        # print(pre_test.shape)
        # metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1), digits=4, output_dict=True)
        print(metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1)))


class LSTM(object):

    def __init__(self):
        self.config = Config()
        self.pre = preprocesser()

    def model(self):
        seq_length = self.config.get("LSTM", "seq_length")
        num_classes = self.config.get("LSTM", "num_classes")
        vocab_size = self.config.get("LSTM", "vocab_size")


        model_input = keras.layers.Input((seq_length))
        embedding = keras.layers.Embedding(vocab_size+1, 256, input_length=seq_length)(model_input)
        LSTM = keras.layers.LSTM(256)(embedding)
        FC1 = keras.layers.Dense(256, activation="relu")(LSTM)
        droped = keras.layers.Dropout(0.5)(FC1)
        FC2 = keras.layers.Dense(num_classes, activation="softmax")(droped)

        model = keras.models.Model(inputs=model_input, outputs=FC2)

        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=["accuracy"])
        model.summary()
        return model

    def train(self, epochs):
        trainingSet_path = self.config.get("data_path", "trainingSet_path")
        valSet_path = self.config.get("data_path", "valSet_path")
        seq_length = self.config.get("LSTM", "seq_length")
        model_save_path = self.config.get("result", "LSTM_model_path")
        batch_size = self.config.get("LSTM", "batch_size")

        model = self.model()

        x_train, y_train = self.pre.word2idx(trainingSet_path, max_length=seq_length)
        x_val, y_val = self.pre.word2idx(valSet_path, max_length=seq_length)

        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      validation_data=(x_val, y_val),
                      epochs=1)
            model.save(model_save_path, overwrite=True)

    def test(self):
        model_save_path = self.config.get("result", "LSTM_model_path")
        testingSet_path = self.config.get("data_path", "testingSet_path")
        seq_length = self.config.get("LSTM", "seq_length")


        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()

        x_test, y_test = self.pre.word2idx(testingSet_path, max_length=seq_length)
        pre_test = model.predict(x_test)

        # metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1), digits=4, output_dict=True)
        print(metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1)))



if __name__ == '__main__':
    test = TextCNN()
    # test.train(3)
    test.test()

    # LSTMTest = LSTM()
    # LSTMTest.train(3)
    # LSTMTest.test()