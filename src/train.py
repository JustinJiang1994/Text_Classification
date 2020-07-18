# -*- coding: utf-8 -*-

"""
Created on 2020-07-19 01:32
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import tensorflow.keras as keras
import numpy as np
from sklearn import metrics
import os

from preprocess import preprocesser
from config import Config
from model import TextCNN


if __name__ == '__main__':
    CNN_model = TextCNN()
    CNN_model.train(3)