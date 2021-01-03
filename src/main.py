# -*- coding: utf-8 -*-

import tensorflow.keras as keras
import numpy as np
from sklearn import metrics
import os

from preprocess import preprocesser
from config import Config
from model import TextCNN
from model import LSTM

np.random.seed(42)

if __name__ == '__main__':
    CNN_model = TextCNN()
    CNN_model.train(5)
    CNN_model.test()
    # LSTM_MODEL = LSTM()
    # LSTM_MODEL.train(5)
    # LSTM_MODEL.test()