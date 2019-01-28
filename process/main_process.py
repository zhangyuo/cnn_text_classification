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
from process.data_process import *
from config.config import *
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
from tool.logger import logger
from tool.text_classify import TextCNN
import csv


def train():
    """
    model train process
    :return:
    """
    logger.info('Loading train data...')
    # Load train data
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
                    checkpoint_every=checkpoint_every)

    model.build_graph()
    model.train(vocab_processor, x_train, y_train, x_dev, y_dev)


def test():
    """
    model test process
    :return:
    """
    logger.info('Loading test data...')
    # Load test data
    x_text, y = load_data_and_labels(TEST_DATA_PATH_NEG, TEST_DATA_PATH_POS)
    # Load vocabulary
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore('../process/runs/1548399694/vocab')
    x_test = np.array(list(vocab_processor.transform(x_text)))
    y_test = y
    logger.info('test data: {}'.format(len(x_test)))
    # Load train model
    # ckpt_file = tf.train.latest_checkpoint('..\\process\\runs\\1548399694\\checkpoints\\')
    # logger.info('model path is %s' % ckpt_file)

    # testing
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    graph = tf.Graph()
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph(
                "{}.meta".format("..\\process\\runs\\1548399694\\checkpoints\\model-2000"))
            saver.restore(sess, "..\\process\\runs\\1548399694\\checkpoints\\model-2000")

            # Get the placeholders from the graph by name
            # a = graph.get_operations()
            # for i in a:
            #     print(i)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(np.argmax(y_test, 1) == all_predictions))
        logger.info("Total number of test examples: {}".format(len(y_test)))
        logger.info("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    # Save the test result to a csv
    predictions_human_readable = np.column_stack((np.array(x_text), all_predictions))
    output_path = "../data/test_data/prediction.csv"
    print("Saving evaluation to {0}".format(output_path))
    with open(output_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)


if __name__ == '__main__':
    train()
    # test()
