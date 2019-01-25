#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/9 14:55
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : main_process.py
@Software: PyCharm
@Github  : zhangyuo
"""
from process.data_process import load_data_and_labels
from config.config import *
from tensorflow.contrib import learn
import numpy as np
from tool.logger import logger
from tool.text_classify import TextCNN


def train():
    """
    model train process
    :return:
    """
    logger.info('Loading data...')
    # Load data
    x_text, y = load_data_and_labels(TRAIN_DATA_PATH_POS, TRAIN_DATA_PATH_NEG)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_shuffled, y_shuffled
    logger.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    model = TextCNN(sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    dropout_keep=dropout,
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_dim=embedding_dim,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    l2_reg_lambda=l2_reg_lambda,
                    optimizer=optimizer,
                    lr=lr,
                    grad_clip=grad_clip,
                    num_checkpoints=num_checkpoints,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    evaluate_every=evaluate_every,
                    checkpoint_every=checkpoint_every
                    )

    model.build_graph()
    model.train(vocab_processor, x_train, y_train, x_dev, y_dev)


if __name__ == '__main__':
    train()
