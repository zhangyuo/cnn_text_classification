#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/9 15:55
@Author  : Zhangyu
@Email   : zhangycqupt@163.com
@File    : text_classify.py
@Software: PyCharm
@Github  : zhangyuo
"""
import tensorflow as tf
import os
import datetime
import time
from tool.logger import logger
from process.data_process import batch_iter


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 dropout_keep,
                 vocab_size,
                 embedding_dim,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda,
                 optimizer,
                 lr,
                 grad_clip,
                 num_checkpoints=5,
                 batch_size=128,
                 num_epochs=200,
                 evaluate_every=100,
                 checkpoint_every=100):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.optimizer = optimizer
        self.lr = lr
        self.grad_clip = grad_clip
        self.num_checkpoints = num_checkpoints
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every

    def build_graph(self):
        self.add_placeholders()
        self.regular_loss_op()
        self.embedding_layer_op()
        self.conv_pool_layer_op()
        self.loss_op()
        self.accuracy_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        """
        Placeholders for input, output and dropout
        :return:
        """
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def regular_loss_op(self):
        """
        Keeping track of l2 regularization loss (optional)
        :return:
        """
        self.l2_loss = tf.constant(0.0)

    def embedding_layer_op(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            _word_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                trainable=False,
                name="_word_embeddings")
            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                          ids=self.input_x,
                                                          name='word_embeddings')
            self.word_embeddings_expanded = tf.expand_dims(input=self.word_embeddings,
                                                           axis=-1,
                                                           name='word_embeddings_expanded')

    def conv_pool_layer_op(self):
        """
        Create a convolution + maxpool layer for each filter size
        :return:
        """
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.word_embeddings_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def loss_op(self):
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def accuracy_op(self):
        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def trainstep_op(self):
        with tf.name_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

            self.grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v] for g, v in
                                   self.grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        """
        Initialize all variables
        :return:
        """
        self.init_op = tf.global_variables_initializer()

    def train(self, vocab_processor, x_train, y_train, x_dev, y_dev):
        """
        model train process
        :return:
        """
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

        with tf.Session() as sess:
            sess.run(self.init_op)
            self._add_summary(sess, vocab_processor)
            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.train_step(sess, x_batch, y_batch)
                current_step = tf.train.global_step(sess, self.global_step)
                if current_step % self.evaluate_every == 0:
                    logger.info("Evaluation:")
                    self.dev_step(sess, x_dev, y_dev)
                    logger.info("")
                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, self.checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {}\n".format(path))

    def train_step(self, sess, x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout: self.dropout_keep_prob
        }
        # scores, predictions = sess.run([self.scores, self.predictions], feed_dict)
        _, step, summaries, loss, accuracy = sess.run(
            [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def dev_step(self, sess, x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout: 1.0
        }
        step, summaries, loss, accuracy = sess.run([self.global_step, self.dev_summary_op, self.loss, self.accuracy],
                                                   feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.dev_summary_writer.add_summary(summaries, step)

    ###########################private func##############################
    def _add_summary(self, sess, vocab_processor):
        """
        Tesorboard 图形化展示
        :param sess:
        :return:
        """
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
