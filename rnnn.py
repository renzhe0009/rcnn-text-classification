# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:03:10 2016

@author: Zhang
"""

import sys
import os
import datetime
import time
import numpy as np
import tensorflow as tf
#from tensorflow.models.rnn import rnn_cell
#from tensorflow.models.rnn import rnn as tf_rnn
from sklearn.metrics import classification_report

sys.path.append(os.pardir)
from utils.mixins import NNMixin, TrainMixin
from utils import ymr_data

# Parameters
# ==================================================

# Model Hyperparameters
SENTENCE_LENGTH_PADDED = NUM_STEPS = 256
HIDDEN_DIM = 128
EMBEDDING_SIZE = 128

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 128
EVALUATE_EVERY = 16
NUM_CLASSES = 2

train_x, train_y, dev_x, dev_y, test_x, test_y = ymr_data.generate_dataset(fixed_length=SENTENCE_LENGTH_PADDED)
VOCABULARY_SIZE = max(train_x.max(), dev_x.max(), test_x.max()) + 1
print("\ntrain/dev/test size: {:d}/{:d}/{:d}\n".format(len(train_y), len(dev_y), len(test_y)))

class CharRNN(object, NNMixin, TrainMixin):
    def __init__(
      self, vocabulary_size, sequence_length, batch_size, num_classes,
      embedding_size=128, hidden_dim=256, cell=None, num_layers=3, loss="linear_gain"):

        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length])
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes])

        if not cell:
            # Standard cell: Stacked LSTM
            first_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
            # first_cell = rnn_cell.LSTMCell(hidden_dim, embedding_size, use_peepholes=True)
            # next_cell = rnn_cell.LSTMCell(hidden_dim, hidden_dim, use_peepholes=True)
            # self.cell = rnn_cell.MultiRNNCell([first_cell] + [next_cell] * (num_layers - 1))
            self.cell = first_cell

        with tf.variable_scope("embedding"):
            self.embedded_chars = self._build_embedding([vocabulary_size, embedding_size], self.input_x)

        with tf.variable_scope("rnn") as scope:
            self.initial_state = tf.placeholder(tf.float32, [batch_size, self.cell.state_size])
            # self.initial_state = tf.Variable(tf.zeros([batch_size, self.cell.state_size]))
#             self.outputs = []
#             self.states = [self.initial_state]
#             for i in range(sequence_length):
#                 if i > 0:
#                     scope.reuse_variables()
#                 new_output, new_state = self.cell(self.embedded_chars[:, i, :], self.states[-1])
#                 self.outputs.append(new_output)
#                 self.states.append(new_state)
#             self.final_state = self.states[-1]
#             self.final_output = self.outputs[-1]
            item_list = [tf.squeeze(x) for x in tf.split(1, sequence_length, self.embedded_chars)]
            self.outputs, self.states = tf.rnn.rnn(self.cell, item_list, initial_state=self.initial_state)
            self.final_state = self.states[-1]
            self.final_output = self.outputs[-1]

        with tf.variable_scope("softmax") as scope:
            self.ys = []
            for i, o in enumerate(self.outputs):
                if i > 0:
                    scope.reuse_variables()
                y = self._build_softmax([hidden_dim, num_classes], o)
                self.ys.append(y)
            self.y = self.ys[-1]
            self.predictions = tf.argmax(self.y, 1)

        with tf.variable_scope("loss"):
#           if loss == "linear_gain":
            # Loss with linear gain. We output at each time step and multiply losses with a linspace
            # Because we have more gradients this can result in faster learning
            self.anneal_factors = tf.placeholder(tf.float32, sequence_length)
            annealed_losses = self._build_annealed_losses(self.ys, self.input_y, self.anneal_factors)
            self.loss = tf.reduce_sum(annealed_losses) / batch_size
            # self.mean_loss = tf.reduce_mean(annealed_losses)
#             elif loss == "last":
#                 # Standard loss, only last output is considered
#                 self.loss = self._build_total_ce_loss(self.ys[-1], self.input_y)
#                 self.mean_loss = self._build_mean_ce_loss(self.ys[-1], self.input_y)
#             self.loss = self._build_total_ce_loss(self.y, self.input_y) / batch_size
#             self.mean_loss = self._build_mean_ce_loss(self.y, self.input_y)                

        # Summaries
        total_loss_summary = tf.scalar_summary("loss", self.loss)
        # mean_loss_summary = tf.scalar_summary("mean loss", self.mean_loss)
        accuracy_summmary = tf.scalar_summary("accuracy", self._build_accuracy(self.y, self.input_y))
        # self.summaries = tf.merge_all_summaries()

    def _build_annealed_losses(self, outputs, labels, anneal_factors):
        sequence_length = len(outputs)
        packed_outputs = tf.pack(outputs)
        tiled_labels = tf.pack([labels for i in range(sequence_length)])
        accumulated_losses = -tf.reduce_sum(tiled_labels * tf.log(packed_outputs), [1, 2])
        annealed_losses = tf.mul(anneal_factors, tf.concat(0, accumulated_losses))
        return annealed_losses
        
        
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Instantiate our model
        rnn = CharRNN(VOCABULARY_SIZE, NUM_STEPS, BATCH_SIZE, 2,
                      embedding_size=EMBEDDING_SIZE)

        # Generate input batches (using tensorflow)
        with tf.variable_scope("input"):
            placeholder_x = tf.placeholder(tf.int32, train_x.shape)
            placeholder_y = tf.placeholder(tf.float32, train_y.shape)
            train_x_var = tf.Variable(placeholder_x, trainable=False, collections=[])
            train_y_var = tf.Variable(placeholder_y, trainable=False, collections=[])
            x_slice, y_slice = tf.train.slice_input_producer([train_x_var, train_y_var], num_epochs=NUM_EPOCHS)
            x_batch, y_batch = tf.train.batch([x_slice, y_slice], batch_size=BATCH_SIZE)

        # Define Training procedure
        out_dir = os.path.join(os.path.curdir, "runs", str(int(time.time())))
        global_step = tf.Variable(0, name="global_step")
        optimizer = tf.train.AdamOptimizer(1e-4)
        # optimizer = tf.train.GradientDescentOptimizer(1e-2)
        # Clip the gradients
        tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(rnn.loss, tvars), 5)
        grads = tf.gradients(rnn.loss, tvars)
        for g, v in zip(grads, tvars):
            if g is not None:
                tf.histogram_summary("{}/grad".format(v.name), g)
                tf.scalar_summary("{}/grad-sparsity".format(v.name), tf.nn.zero_fraction(g))
                    
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        summary_op = tf.merge_all_summaries()
        
        # Summaries
        train_summary_dir = os.path.abspath(os.path.join(out_dir, "summaries", "train"))
        print(train_summary_dir)
        train_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)            
        
        # Generate train and eval seps
#         train_step = rnn.build_train_step(
#             out_dir, train_op, global_step, rnn.summaries, ops=[rnn.final_state], save_every=8, sess=sess)
#         eval_step = rnn.build_eval_step(out_dir, rnn.predictions, global_step, rnn.summaries, sess=sess)

        # Initialize variables and input data
        sess.run(tf.initialize_all_variables())
        sess.run(
            [train_x_var.initializer, train_y_var.initializer],
            {placeholder_x: train_x, placeholder_y: train_y})

        # Initialize queues
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Print model parameters
        # rnn.print_parameters()
        
        def rolled_batches(x_batch, y_batch, gains):
            num_unrolls = SENTENCE_LENGTH_PADDED/NUM_STEPS
            x_unrolls = np.split(x_batch, num_unrolls, 1)
            gain_unrolls = np.split(gains, num_unrolls)
            for x_unroll, gain_unroll in zip(x_unrolls, gain_unrolls):
                feed_dict = {
                    rnn.input_x: x_unroll,
                    rnn.input_y: y_batch,
                    rnn.anneal_factors: gain_unroll
                }
                yield feed_dict
                
        def eval_dev(dev_x, dev_y):
            drop_num_elements = len(dev_y) % BATCH_SIZE
            if drop_num_elements > 0:
                dev_x_ = dev_x[:-drop_num_elements]
                dev_y_ = dev_y[:-drop_num_elements]
            nbatches = len(dev_y)/BATCH_SIZE
            predictions = []
            # For each batch...
            for batch_x, batch_y in zip(np.split(dev_x_, nbatches), np.split(dev_y_, nbatches)):
                gains = np.zeros(SENTENCE_LENGTH_PADDED) # Not used
                state = np.zeros(rnn.initial_state.get_shape().as_list())
                # For each unroll step...
                for feed_dict in rolled_batches(batch_x, batch_y, gains):
                    feed_dict[rnn.initial_state] = state
                    batch_predictions, state = sess.run([rnn.predictions, rnn.final_state], feed_dict)
                predictions = np.append(predictions, batch_predictions)
            print(classification_report(np.argmax(dev_y_, axis=1), predictions))
            
        def train_step(batch_x, batch_y):
            state = np.zeros(rnn.initial_state.get_shape().as_list())
            # gains = np.linspace(0.0, 1.0, SENTENCE_LENGTH_PADDED)
            # Only consider the last loss
            gains = np.zeros(SENTENCE_LENGTH_PADDED)
            gains[-1] = 1.0
            # We unroll the nework several times and pass the state
            for feed_dict in rolled_batches(batch_x, batch_y, gains):
                feed_dict[rnn.initial_state] = state
                _, state, loss, global_step_, summaries_ = sess.run(
                    [train_op, rnn.final_state, rnn.loss, global_step, summary_op],
                    feed_dict=feed_dict)
            train_writer.add_summary(summaries_, global_step_)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, global_step_, loss))
            
        # Repeat until we're done (the input queue throws an error)...
        try:
            while not coord.should_stop():
                batch_x = x_batch.eval()
                batch_y = y_batch.eval()
                train_step(batch_x, batch_y)
                if global_step.eval() % EVALUATE_EVERY == 0:
                    # eval_dev(dev_x, dev_y)
                    pass
                    
        except tf.errors.OutOfRangeError:
            print("Yay, training done!")
            eval_step({rnn.input_x: dev_x, rnn.input_y: dev_y})
        finally:
            coord.request_stop()
        coord.join(threads)