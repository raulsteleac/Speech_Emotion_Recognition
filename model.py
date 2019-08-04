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

from reader import Data_Producer

class SER_Data_Producer(object):
      def __init__(self, config):
            self.dp = Data_Producer(config)

      def import_data(self, session):
            (self.train_inputs, self.train_targets), (self.test_inputs, self.test_targets), (self.train_length, self.test_length) = self.dp.produce_data(session)
      
      @property
      def train_data(self):
            return self.train_inputs, self.train_targets, self.train_length

      @property
      def test_data(self):
            return self.test_inputs, self.test_targets, self.test_length

class Speech_Emotion_Recognizer(object):
      def __init__(self, inputs, targets, op_length, model_op_name, is_training=False):
            self._inputs = inputs
            self._targets = targets
            self._op_length = op_length
            self._is_training = is_training

            self._learning_rate = 0.003
            self._keep_prob = 0.7
            self.model_op_name = model_op_name
            self.init = tf.random_uniform_initializer(-0.1, 0.1)

      def model(self):
            with tf.variable_scope("Speech_Emotion_Recognizer", reuse = tf.AUTO_REUSE, initializer=self.init):
                  rnn_layer_1 = self.create_LSTM_layer(self._inputs)
                  fully_connected_layer = tf.layers.dense(rnn_layer_1[0], 4, name="Output_Layer")
                  predictions = self.create_attention_layer(fully_connected_layer)

                  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = self._targets, logits = predictions)
                  targets_ = tf.nn.sigmoid(predictions)
                  targets_ = tf.round(targets_)
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
            print("=========Create LSTM Cell")
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, use_peepholes=True)
            if self._is_training and self._keep_prob < 1:
                  cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
            return cell

      def create_LSTM_layer(self, inputs):
            lstm_cell = self.make_lstm_cell(75)
            initial_zero_state = lstm_cell.zero_state(1, tf.float64)

            inputs = tf.expand_dims(inputs, axis=0)
            _, states = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=initial_zero_state)
            return states

      def create_attention_layer(self, frame_predictions):
            W = tf.get_variable("Attention_Weights", dtype=tf.float64, shape=[4, 1])
            b = tf.get_variable("Attention_Bias", dtype=tf.float64, shape=[1])
            
            alpha = tf.matmul(frame_predictions, W) + b
            alpha = tf.nn.softmax(alpha, axis=0)
            return tf.reduce_sum(tf.multiply(frame_predictions, alpha[: tf.newaxis]), axis = 0)

      @property
      def running_ops(self):
            """ RETURNS THE TENSORS THAT NEED TO BE COMPUTED. DEPENDENCIES WILL MAKE ALL THE OTHER NECESSARY TENSORS BE COMPUTED TOO.
            """
            if self._is_training:
                  return {
                      "accuracy": self.accuracy,
                      "optimizer": self.optimizer
                  }
            else:
                  return {
                      "accuracy": self.accuracy
                  }

      def run_model(self, session, writer, merged_summaries):
            """ RUNNING MODEL ON CURRENT CONFIGURATION 
                This method is computing training, validation or testing depending on what model is calling it.
            """
            print("\n %s just started !" % self.model_op_name)
            accuracy = 0.0
            for instance in range(self._op_length):
                  vals = session.run(self.running_ops)
                  accuracy +=  vals["accuracy"]
                  if (self._op_length // 10) != 0 and instance % (self._op_length // 10) == 0:
                        print("-----------> Instance number : %d Current Accuracy : %f" % (instance / (self._op_length // 10),  vals["accuracy"]))
                  writer.add_summary(session.run(merged_summaries))
            print("############### %s Total Accuracy = %lf \n" % (self.model_op_name, (accuracy / self._op_length)))
            
      def debug_print(self, session):
            print(type(self._inputs))
            print(self._inputs)
            print(session.run(self._inputs).shape)
            print(session.run(self._targets).shape)

class NormalConfig(object):
      dir_name = 'EMO-DB'
      data_set_name = 'EMO-DB'
      train_test_slice = 0.8

def main():
      ses = tf.Session()

      ser_dp = SER_Data_Producer(NormalConfig())
      ser_dp.import_data(ses)

      train_inputs, train_targets, train_length = ser_dp.train_data
      test_inputs, test_targets, test_length = ser_dp.test_data

      ser_train_model = Speech_Emotion_Recognizer(train_inputs, train_targets, train_length, "Training", True)
      ser_test_model  = Speech_Emotion_Recognizer(test_inputs, test_targets, test_length, "Testing")

      ser_train_model.debug_print(ses)
      ser_train_model.model()
      ser_test_model.model()

      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()

      ser_train_model.initialize_variables(ses)
      epochs = 5
      for epoch in range(epochs):
            print("\n-----------> Epoch %d" % epoch)
            ser_train_model.run_model(ses, writer, merged_summaries)
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      ser_test_model.run_model(ses, writer, merged_summaries)

      ses.close()

if __name__ == '__main__':
      main()

#%%
