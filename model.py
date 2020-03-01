#%%
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'Speech_Emotion_Recognition'))
    print(os.getcwd())
except:
    pass

import numpy as np
import tensorflow as tf

from feature_extractors.hand_crafted_data_producers import Data_Producer_Hand_Crafted_Train_Test
from feature_extractors.hand_crafted_data_producers import Data_Producer_Hand_Crafted_Inference
from feature_extractors.end_to_end_data_producers import Data_Producer_End_to_End_Train_Test
from feature_extractors.end_to_end_data_producers import Data_Producer_End_to_End_Inference
from feature_extractors.online_inference_extractor import Online_Data_Producer_End_to_End_Inference
from util import *
from PyQt5 import QtCore

class SER_Data_Producer(object):
      def __init__(self, config, flag_end_to_end=1):
            if flag_end_to_end:
                  self.dp = Data_Producer_End_to_End_Train_Test(config)
            else:
                  self.dp = Data_Producer_Hand_Crafted_Train_Test(config)

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
      def __init__(self, model_op_name = "", is_training=False, is_inference=False, flag_end_to_end = 1):
            self._is_training = is_training
            self._is_inference = is_inference

            self._hidden_size = 256 if flag_end_to_end==0 else 1024 
            self._emotion_nr = 4
            self._learning_rate = 0.001
            self._keep_prob = 0.8
            self.model_op_name = model_op_name
            self.init = tf.random_normal_initializer(-0.1, 0.1, seed = 17)

      def set_inputs_targets_length(self, inputs, targets=None, op_length=1):
            self._inputs = inputs
            self._targets = targets
            self._op_length = op_length
            
      def model(self):
            """ MAIN FUNCTION OF THE CLASS, RESPONSIBLE FOR CREATING THE MODEL
                  -The weights, and all the other necessary parameters for all the models,
                        will be share using the tf.virtual_scope.
            """
            with tf.variable_scope("Speech_Emotion_Recognizer", reuse=tf.AUTO_REUSE, initializer=self.init):
                  # self._inputs = batch_normalization(self._inputs)
                  rnn_layer = self.create_LSTM_layer(self._inputs, self._hidden_size, "rnn_layer1")
                  # rnn_layer = batch_normalization(rnn_layer)

                  attention_layer_output = self.create_attention_layer(rnn_layer, self._hidden_size)
                  predictions_1 = tf.layers.dense(attention_layer_output, self._emotion_nr, name="Output_Layer")
                  predictions = tf.reduce_sum(predictions_1, axis=0)

                  targets_raw_ = tf.nn.softmax(predictions, axis=0)

                  # targets_ = tf.cast(tf.equal(targets_raw_, tf.reduce_max(targets_raw_)), tf.float32)
                  targets_ = tf.round(targets_raw_)

                  if self._is_inference:
                        self.predictions_raw = targets_
                        self.predictions = targets_raw_
                        return
                  
                  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = self._targets, logits = targets_raw_)

                  self._label_pred = tf.argmax(targets_)
                  self._label_true = tf.argmax(self._targets)

                  # self.accuracy = tf.cast(tf.equal(self._label_pred, self._label_true), tf.float32)
                  is_correct = tf.equal(targets_, self._targets)
                  self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                  if not self._is_training:
                        return

                  tf.summary.scalar('Accuracy', self.accuracy)
                  adam_opt = tf.train.AdamOptimizer(self._learning_rate)
                  self.optimizer = adam_opt.minimize(cross_entropy)

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
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, use_peepholes=True, activation = tf.nn.elu)
            if self._is_training and self._keep_prob < 1:
                  cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
            return cell

      def create_LSTM_layer(self, inputs, hidden_size, name=None):
            """ CREATES A RNN LAYER BASED ON A LSTM CELL AND A INITIAL ZERO STATE
                  -Arguments:
                        hidden_size: the size of the internal layers of the RNN cell
                  -Returns:
                        states[0]:the outputs of each state in the RNN sequence
            """
            with tf.variable_scope(name):
                  lstm_cells = [self.make_lstm_cell(hidden_size) for _ in range(1)]
                  multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
                  initial_zero_state = multi_cell.zero_state(1, tf.float32)
                  inputs = tf.expand_dims(inputs, axis=0)
                  outputs, states = tf.nn.dynamic_rnn(
                      multi_cell, inputs, initial_state=initial_zero_state)
            return outputs[0]

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
                        "label_true": self._label_true
                        }
                  else:
                        return {
                        "accuracy": self.accuracy,
                        "label_pred": self._label_pred,
                        "label_true": self._label_true
                        }

      def run_model(self, session, writer, merged_summaries, files = None, file_to_show=None, thread = None, feed_dict=None):
            """ RUNNING MODEL ON CURRENT CONFIGURATION 
                  This method is computing training, validation or testing depending on what model is calling it.
                  -Arguments:
                        session: the tf.Session() the model is running on
                        writer: the tf.summary.FileWriter used to save the graphs
                        merged_summaries: the summaries use to see the evolution of the model
            """
            print("\n %s just started ! \n" % self.model_op_name)
            self.accuracy_matrix = np.zeros((self._emotion_nr, self._emotion_nr))            
            if not self._is_inference:
                  total_accuracy = 0.0
                  sample_accuray = 0.0
                  print_accuracy = 0.0
                  sample_size = (self._op_length // 10)
                  print_rate = (self._op_length // 10)

                  for instance in range(self._op_length):
                        vals = session.run(self.running_ops)
                        total_accuracy += vals["accuracy"]
                        sample_accuray += vals["accuracy"]
                        print_accuracy += vals["accuracy"]

                        self.accuracy_matrix[vals["label_pred"]][vals["label_true"]] += 1
                        if self._is_training and thread != None and instance != 0 and instance % print_rate == 0:
                              thread.print_accuracy_signal.emit(print_accuracy / print_rate)
                              print_accuracy = 0.0
                        if instance != 0 and instance % sample_size == 0 :
                              if thread == None:
                                    print("-----------> Instance number : %d Current Accuracy : %f" % (instance / sample_size, sample_accuray / sample_size))
                              else:
                                    print("-----------> Instance number : %d Current Accuracy : %f" % (instance / sample_size, sample_accuray / sample_size))
                                    thread.print_stats.emit(str("-----------> Instance number : " + str(instance / sample_size) + " Current Accuracy: " + str(sample_accuray / sample_size)))
                              sample_accuray = 0.0
                        if thread != None and thread.stopFlag == True:
                              return
                  if thread == None:
                        print("############### %s Total Accurac y = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length)))      
                  else:
                        print("############### %s Total Accuracy = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length)))      
                        thread.print_stats.emit(str("############### %s Total Accuracy = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length))))
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
            print(self.accuracy_matrix)

      def debug_print(self, session):
            print(type(self._inputs))
            print(self._inputs)
            print(self._targets)
            print(session.run(self._inputs).shape)
            print(session.run(self._targets).shape)

      def create_saver(self):
            self.saver = tf.train.Saver()

      def save_model(self, ses, path):
            self.saver.save(ses, path)

      def restore_model(self, ses, path):
            self.saver.restore(ses, path)
#%%
def main(thread=None, epochs=10, id_config=1, flag_end_to_end = 1):
      tf.reset_default_graph()
      ses = tf.Session()

      ser_dp = SER_Data_Producer(select_config(id_config), flag_end_to_end=flag_end_to_end)
      ser_dp.import_data(ses)

      train_inputs, train_targets, train_length = ser_dp.train_data
      test_inputs, test_targets, test_length = ser_dp.test_data

      ser_train_model = Speech_Emotion_Recognizer( "Training", True, flag_end_to_end=flag_end_to_end)
      ser_test_model  = Speech_Emotion_Recognizer( "Testing",  flag_end_to_end=flag_end_to_end)

      ser_train_model.set_inputs_targets_length(train_inputs, train_targets, train_length)
      ser_test_model.set_inputs_targets_length(test_inputs, test_targets, test_length)

      ser_train_model.model()
      ser_test_model.model()

      ser_train_model.create_saver()

      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()

      ser_train_model.initialize_variables(ses)
      # print_accuracy_graph(thread, 0)
      for epoch in range(epochs):
            print("-----------> Epoch " + str(epoch))
            ser_train_model.run_model(ses, writer, merged_summaries, thread=thread)
      ser_train_model.save_model(ses, "./model/model.ckpt")
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      ser_test_model.run_model(ses, writer, merged_summaries, thread=thread)

      # clear_accuracy_vals()
      ses.close()
#%%
def init_inference_model(flag_end_to_end=1):
      ses = tf.Session()

      if flag_end_to_end:
            ser_dp_inference = Data_Producer_End_to_End_Inference(
                Inference_Config())
      else:
            ser_dp_inference = Data_Producer_Hand_Crafted_Inference(
                Inference_Config())

      ser_inference_model = Speech_Emotion_Recognizer(
          model_op_name="Inference", is_training=False, is_inference=True)

      infr_inputs, inference_length, files = ser_dp_inference.produce_data(ses)
      ser_inference_model.set_inputs_targets_length(
          inputs=infr_inputs, op_length=inference_length)

      ser_inference_model.model()
      ser_inference_model.create_saver()
      ser_inference_model.restore_model(ses, "./model/model.ckpt")

      ser_inference_model.model()
      return ses, ser_inference_model, files

def close_inference_model(ses):
      ses.close()


def inference(ses, ser_inference_model, files, file_to_show):
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()
      list_vars = ser_inference_model.run_model(ses, writer, merged_summaries, files, file_to_show)
      return list_vars

infr_inputs = tf.placeholder(tf.float32, (None, 1024))
inference_length = tf.placeholder(tf.float32, None)      
def init_online_model():
      global infr_inputs, inference_length
      ses = tf.Session()     

      ser_inference_model = Speech_Emotion_Recognizer(
      model_op_name="Online", is_training=False, is_inference=True)
      ser_inference_model.set_inputs_targets_length(inputs=infr_inputs, op_length=inference_length)

      ser_inference_model.model()
      ser_inference_model.create_saver()
      ser_inference_model.restore_model(ses, "./model/model.ckpt")

      ser_inference_model.model()
      return ses, ser_inference_model


def online(ses, ser_inference_model, frames, org_rt):
      global infr_inputs, inference_length
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()
      ser_dp_online = Online_Data_Producer_End_to_End_Inference()
      online_inputs_, online_length_ = ser_dp_online.produce_data(ses, frames, org_rt)
      list_vars = ser_inference_model.run_model(ses, writer, merged_summaries, feed_dict={
                                                infr_inputs: ses.run(online_inputs_), inference_length: online_length_})
      return list_vars

#%%

if __name__ == "__main__":
    main(epochs=30)


#%%


# def inference_work(flag_end_to_end=1):
#       ses = tf.Session()

#       if flag_end_to_end:
#             ser_dp_inference = Data_Producer_End_to_End_Inference(
#                 Inference_Config())
#       else:
#             ser_dp_inference = Data_Producer_Hand_Crafted_Inference(
#                 Inference_Config())

#       ser_inference_model = Speech_Emotion_Recognizer(
#           model_op_name="Inference", is_training=False, is_inference=True)
#       ser_inference_model.create_saver()
#       ser_inference_model.restore_model(ses, "./model/model.ckpt")
#       infr_inputs, inference_length, files = ser_dp_inference.produce_data(ses)

#       ser_inference_model.set_inputs_targets_length(
#           inputs=infr_inputs, op_length=inference_length)

#       writer = tf.summary.FileWriter('./graphs', ses.graph)
#       merged_summaries = tf.summary.merge_all()

#       ser_inference_model.model()
#       ser_inference_model.run_model(ses, writer, merged_summaries, files)

#       ses.close()


# inference_work()


# #%%


#%%
