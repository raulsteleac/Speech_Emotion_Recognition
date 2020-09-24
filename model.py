#%%
import os

import numpy as np
from feature_extractors.hand_crafted_data_producers import Data_Producer_Hand_Crafted_Train_Test
from feature_extractors.end_to_end_data_producers import Data_Producer_End_to_End_Train_Test
from util import *
from PyQt5 import QtCore

class SER_Data_Producer(object):
      def __init__(self, config, train_ratio, flag_end_to_end=1, thread=None):
            if flag_end_to_end:
                  self.dp = Data_Producer_End_to_End_Train_Test(config, train_ratio, thread)
            else:
                  self.dp = Data_Producer_Hand_Crafted_Train_Test(config, train_ratio, thread)

      def import_data(self, session): 
            """ CALLS THE PRODUCE_DATA FUNCTION OF THE DATA_PRODUCER
                    -Arguments:
                        session: the tf.Session() the model is running on
            """
            (self.train_inputs, self.train_targets), self.train_length = self.dp.produce_data_train(session)
            (self.test_inputs, self.test_targets), self.test_length = self.dp.produce_data_test(session)
      
      @property
      def train_data(self):
            """ RETURNS NECESSARY INFORMATION FOR THE TRAINING PHASE
            """
            return self.train_inputs, self.train_targets, self.train_length

      @property
      def test_data(self):
            """ RETURNS NECESSARY INFORMATION FOR THE TESTING PHASE
            """
            return self.test_inputs, self.test_targets, self.test_length

class Speech_Emotion_Recognizer(object):
      def __init__(self, model_op_name = "", keep_prob= 1, learning_rate=0.0001, is_training=False, is_inference=False, flag_end_to_end = 1):
            self._is_training = is_training
            self._is_inference = is_inference

            self._hidden_size = 75 if flag_end_to_end==0 else 256 
            self._emotion_number = 4
            self._learning_rate = learning_rate
            self._keep_prob = keep_prob
            self.model_op_name = model_op_name
            self.current_examp = 0
            self.init = tf.glorot_normal_initializer()

      def set_inputs_targets_length(self, inputs, targets=None, op_length=1):
            self._inputs = inputs
            self._targets = targets
            self._op_length = op_length
            self._init_op_length = op_length

      def refresh_current_examp(self):
            self.current_examp = 0
            
      def model(self):
            """ MAIN FUNCTION OF THE CLASS, RESPONSIBLE FOR CREATING THE MODEL
                  -The weights, and all the other necessary parameters for all the models,
                        will be share using the tf.virtual_scope.
            """
            print(self._learning_rate)
            with tf.variable_scope("Speech_Emotion_Recognizer", reuse=tf.AUTO_REUSE, initializer=self.init):
                  rnn_layer = self.create_LSTM_layer(self._inputs, self._hidden_size, "rnn_layer1")

                  attention_layer_output = self.create_attention_layer(rnn_layer, self._hidden_size * 2)
                  predictions_1 = tf.layers.dense(attention_layer_output, self._emotion_number, name="Output_Layer")
                  predictions = tf.reduce_sum(predictions_1, axis=0)

                  targets_raw_ = tf.nn.softmax(predictions, axis=0)

                  targets_ = tf.cast(tf.equal(targets_raw_, tf.reduce_max(targets_raw_)), tf.float32)

                  if self._is_inference:
                        self.predictions_raw = targets_
                        self.predictions = targets_raw_
                        return
                  
                  self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._targets, logits=predictions)

                  self._label_pred = tf.argmax(targets_raw_)
                  self._label_true = tf.argmax(self._targets)

                  self.accuracy = tf.cast(tf.equal(self._label_pred, self._label_true), tf.float32)
                  if not self._is_training:
                        return

                  tf.summary.scalar('Accuracy', self.accuracy)
                  adam_opt = tf.train.AdamOptimizer(self._learning_rate)
                  self.optimizer = adam_opt.minimize(self.cross_entropy)

      def initialize_variables(self, session):
            session.run(tf.global_variables_initializer())

      def make_lstm_cell(self, hidden_size):
            """ CREATES THE LSTM CELL BASED ON THE REQUESTE HIDDEN LAYER SIZE AND KEEP PROBABILITY
                  -Arguments:
                        hidden_size: the size of the internal layers of the RNN cell
                  -Returns:
                        The created lstm cell using tf.contrib.rnn.LSTMCell. The weights of the hidden layer
                        of this LSTM cell are shared in the model's variable_scope
            """
            print("=========Create LSTM Cell")
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, use_peepholes=True, initializer=tf.glorot_uniform_initializer())
            if self._is_training and self._keep_prob < 1:
                  cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self._keep_prob, output_keep_prob=self._keep_prob, variational_recurrent=True, dtype=tf.float32, input_size=hidden_size)
            return cell

      def create_LSTM_layer(self, inputs, hidden_size, name=None):
            """ CREATES A RNN LAYER BASED ON A LSTM CELL AND A INITIAL ZERO STATE
                  -Arguments:
                        hidden_size: the size of the internal layers of the RNN cell
                  -Returns:
                        states[0]:the outputs of each state in the RNN sequence
            """
            with tf.variable_scope(name):
                  lstm_cells_fw = [self.make_lstm_cell(hidden_size) for _ in range(2)]
                  lstm_cells_bw = [self.make_lstm_cell(hidden_size) for _ in range(2)]
                  multi_cell_fw = tf.contrib.rnn.MultiRNNCell(lstm_cells_fw, state_is_tuple=True)
                  multi_cell_bw = tf.contrib.rnn.MultiRNNCell(lstm_cells_bw, state_is_tuple=True)
                  initial_zero_state_fw = multi_cell_fw.zero_state(1, tf.float32)
                  initial_zero_state_bw = multi_cell_bw.zero_state(1, tf.float32)
                  inputs = tf.expand_dims(inputs, axis=0)
                  outputs, _ = tf.nn.bidirectional_dynamic_rnn(multi_cell_fw, multi_cell_bw, inputs, initial_state_fw=initial_zero_state_fw, initial_state_bw=initial_zero_state_bw)
            return tf.concat(outputs, 2)[0]

      def create_attention_layer(self, frame_predictions, weigths_dim):
            """ CREATES THE ATTENTION LAYER IN ORDER TO OBTAIN A WEIGHTED POOL LAYER BASED ON THE
                EMOTION IN EACH FRAME
                  -Arguments:
                        frame_predictions: the outputs of the model for each frame
                        weights_dim: size of the attenention layer's weigth
                  -Returns:
                        The weighted sum of all the emotion predictions of all frames  
            """
            W = tf.get_variable("Attention_Weights", dtype=tf.float32, shape=[weigths_dim, 1])
            b = tf.get_variable("Attention_Bias", dtype=tf.float32, shape=[1])
            
            alpha = tf.matmul(frame_predictions, W) + b
            alpha = tf.nn.softmax(alpha, axis=0)
            return tf.multiply(frame_predictions, alpha[: tf.newaxis])

      @property
      def running_ops(self):
            """ RETURNS THE TENSORS THAT NEED TO BE COMPUTED. DEPENDENCIES WILL MAKE ALL THE OTHER NECESSARY TENSORS BE COMPUTED TOO.
            """
            if self._is_inference:
                  return {
                      "predictions": self.predictions,
                      "predictions_raw": self.predictions
                  }
            else:
                  if self._is_training:
                        return {
                        "accuracy": self.accuracy,
                        "optimizer": self.optimizer,
                        "label_pred": self._label_pred,
                        "label_true": self._label_true,
                        "cross_entropy": self.cross_entropy
                        }
                  else:
                        return {
                        "accuracy": self.accuracy,
                        "label_pred": self._label_pred,
                        "label_true": self._label_true
                        }

      def init_examples(self, indexes):
            self.examples_dict = dict.fromkeys(indexes, 0)
      
      def get_keys(self):
            return list(self.examples_dict.keys())

      def add_to_input_examples_results(self, index, val):
            self.examples_dict[index] += val

      def update_input_length(self, epochs, epoch, fraction):
            print(fraction)
            self._op_length = self._op_length - int((fraction) * self._init_op_length // (int((epochs-1) // 5)))
      
      def calculate_worst_input_examples(self):
            # self.examples_dict = {k: v for k, v in sorted(self.examples_dict.items(), key=lambda item: item[1])}
            del_keys = self.get_keys()[self._op_length:]
            for key in del_keys:
                  self.examples_dict.pop(key) 
            self.examples_dict = {k: v for k, v in sorted(self.examples_dict.items(), key=lambda item: item[0])}

      def run_model(self, session, writer, merged_summaries, files = None, file_to_show=None, thread = None, feed_dict=None, validation=False):
            """ RUNNING MODEL ON CURRENT CONFIGURATION 
                  This method is computing training, validation or testing depending on what model is calling it.
                  -Arguments:
                        session: the tf.Session() the model is running on
                        writer: the tf.summary.FileWriter used to save the graphs
                        merged_summaries: the summaries use to see the evolution of the model
            """
            if self._is_training and thread != None and thread.stopFlag == True:
                  return 1
            self.accuracy_matrix = np.zeros((self._emotion_number, self._emotion_number))         
            if not self._is_inference:
                  total_accuracy = 0.0
                  sample_accuray = 0.0
                  print_accuracy = 0.0
                  sample_size = (self._op_length // 10)
                  print_rate = (self._op_length // 10)

                  print("indexes_used = %d" % len(get_indexes()))
                  print("op_length = %d" % self._op_length)
                  for instance in range(self._op_length):
                        vals = session.run(self.running_ops)
                        total_accuracy += vals["accuracy"]
                        sample_accuray += vals["accuracy"]
                        print_accuracy += vals["accuracy"]

                        if self._is_training:
                              self.add_to_input_examples_results(self.get_keys()[self.current_examp], vals['cross_entropy'])
                              self.current_examp+=1

                        self.accuracy_matrix[vals["label_pred"]][vals["label_true"]] += 1
                        if self._is_training and thread != None and instance != 0 and instance % print_rate == 0:
                              thread.print_accuracy_signal.emit(print_accuracy / print_rate)
                              print_accuracy = 0.0
                        if instance != 0 and instance % sample_size == 0 :
                              if thread == None:
                                    print("-----------> Instance number : %d Current Accuracy : %f" % (instance / sample_size, sample_accuray / sample_size))
                              else:
                                    print("-----------> Instance number : %d Current accuracy : %f" % (instance / sample_size, sample_accuray / sample_size))
                                    if not validation:
                                          thread.print_stats.emit(str("-----------> Instance number : " + str(instance / sample_size) + " Current Accuracy: " + str(sample_accuray / sample_size)))
                              sample_accuray = 0.0
                        if thread != None and thread.stopFlag == True:
                              return 1  
                  if thread:
                        thread.app_rnning.label_total.setText("Numarul total de intrari = %d" % self._op_length)                      
                  if thread == None:
                        print("############### %s Total Accurac y = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length)))      
                  else:
                        print("############### %s Total Accuracy = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length)))      
                        if not validation:
                              thread.print_stats.emit(str(" +++ %s total accuracy = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length))))
                        else:
                              thread.print_stats.emit(str(" +++ Validation total accuracy = %lf \n" % ((total_accuracy / self._op_length))))
                        thread.print_matrix.emit(self.accuracy_matrix)
            else:
                  predictions = []
                  if files is None:
                        return session.run(self.running_ops, feed_dict)["predictions_raw"]
                  for i in range(self._op_length):
                        vals = session.run(self.running_ops)
                        if files[i] == file_to_show:
                              predictions =  vals["predictions_raw"]
                  return predictions

      def create_saver(self):
            self.saver = tf.train.Saver()

      def save_model(self, ses, path):
            self.saver.save(ses, path)

      def restore_model(self, ses, path):
            self.saver.restore(ses, path)

