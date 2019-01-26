#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/9 15:02
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : config.py
@Software: PyCharm
@Github  : zhangyuo
"""

# train data
TRAIN_DATA_PATH_NEG = '../data/rt-polaritydata/rt-polarity.neg'
TRAIN_DATA_PATH_POS = '../data/rt-polaritydata/rt-polarity.pos'
# test data
TEST_DATA_PATH_NEG = '../data/test_data/rt-polarity.neg'
TEST_DATA_PATH_POS = '../data/test_data/rt-polarity.pos'

# model Hyperparameters
# percentage of the training data to use for validation
dev_sample_percentage = 0.1
# dropout keep_prob
dropout = 0.5
# word embedding dim
embedding_dim = 256
# comma-separated filter sizes
filter_sizes = 3, 4, 5
# number of filters per filter size
num_filters = 128
# l2 regularization lambda
l2_reg_lambda = 0.5
# Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD
optimizer = "Adam"
# learning rate
lr = 1e-3
# gradient clipping
grad_clip = 5.0

# training parameters
# number of checkpoints to store
num_checkpoints = 5
# batch Size
batch_size = 128
# number of training epochs
num_epochs = 200
# evaluate model on dev set after this many steps
evaluate_every = 100
# save model after this many steps
checkpoint_every = 100
