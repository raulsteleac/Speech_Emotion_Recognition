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

from hand_crafted_data_producers import Data_Producer_Hand_Crafted_Train_Test
from hand_crafted_data_producers import Data_Producer_Hand_Crafted_Inference
from end_to_end_data_producers import Data_Producer_End_to_End_Train_Test
from end_to_end_data_producers import Data_Producer_End_to_End_Inference

flag_end_to_end = 0

# from conv_reader import Data_Producer_Train_Test_Auto

class SER_Data_Producer(object):
      def __init__(self, config):
            if flag_end_to_end:
                  self.dp = Data_Producer_End_to_End_Train_Test(config)
            else:
                  self.dp = Data_Producer_Hand_Crafted_Train_Test(config)

      def import_data(self, session): 
            """ CALLS THE PRODUCE_DATA FUNCTION OF THE DATA_PRODUCER
            """
            (self.train_inputs, self.train_targets), (self.test_inputs, self.test_targets), (self.train_length, self.test_length) = self.dp.produce_data(session)
      
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
      def __init__(self, inputs, targets = None, op_length = 1, model_op_name = "", is_training=False, is_inference=False):
            self._inputs = inputs
            self._targets = targets
            self._op_length = op_length
            self._is_training = is_training
            self._is_inference = is_inference

            self._hidden_size = 256
            self._emotion_nr = 5
            self._learning_rate = 0.001
            self._keep_prob = 0.7
            self.model_op_name = model_op_name
            self.init = tf.random_normal_initializer(-0.1, 0.1)

      def model(self):
            """ MAIN FUNCTION OF THE CLASS, RESPONSIBLE FOR CREATING THE MODEL
                The weights, and all the other necessary parameters for all the models,
                will be share using the tf.virtual_scope.
            """
            with tf.variable_scope("Speech_Emotion_Recognizer", reuse = tf.AUTO_REUSE, initializer=self.init):
                  rnn_layer_1 = self.create_LSTM_layer(self._inputs, self._hidden_size)
                  attention_layer_output = self.create_attention_layer(rnn_layer_1, self._hidden_size)
                  predictions = tf.layers.dense(attention_layer_output, self._emotion_nr, name="Output_Layer")
                  predictions = tf.reduce_sum(predictions, axis = 0)

                  targets_raw_ = tf.nn.sigmoid(predictions)
                  targets_ = tf.round(targets_raw_)

                  if self._is_inference:
                        self.predictions_raw = targets_
                        self.predictions = targets_raw_
                        return
                  
                  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = self._targets, logits = predictions)
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
                Returns:
                The created lstm cell using tf.contrib.rnn.LSTMCell. The weights of the hidden layer
                of this LSTM cell are shared in the model's variable_scope
            """
            print("=========Create LSTM Cell")
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, use_peepholes=True)
            if self._is_training and self._keep_prob < 1:
                  cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
            return cell

      def create_LSTM_layer(self, inputs, hidden_size):
            """ CREATES A RNN LAYER BASED ON A LSTM CELL AND A INITIAL ZERO STATE
            """
            lstm_cell = self.make_lstm_cell(hidden_size)
            initial_zero_state = lstm_cell.zero_state(1, tf.float32)

            inputs = tf.expand_dims(inputs, axis=0)
            _, states = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=initial_zero_state)
            return states[0]

      def create_attention_layer(self, frame_predictions, weigths_dim):
            """ CREATES THE ATTENTION LAYER IN ORDER TO OBTAIN A WEIGHTED POOL LAYER BASED ON THE
                EMOTION IN EACH FRAME
                Returns:
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
                        "optimizer": self.optimizer
                        }
                  else:
                        return {
                        "accuracy": self.accuracy
                        }

      def run_model(self, session, writer, merged_summaries, files = None):
            """ RUNNING MODEL ON CURRENT CONFIGURATION 
                This method is computing training, validation or testing depending on what model is calling it.
            """
            print("\n %s just started ! \n" % self.model_op_name)
            if not self._is_inference:
                  total_accuracy = 0.0
                  sample_accuray = 0.0
                  sample_size = (self._op_length // 10)
                  for instance in range(self._op_length):
                        vals = session.run(self.running_ops)
                        total_accuracy += vals["accuracy"]
                        sample_accuray += vals["accuracy"]
                        if instance != 0 and instance % sample_size == 0:
                              print("-----------> Instance number : %d Current Accuracy : %f" % (instance / sample_size, sample_accuray / sample_size))
                              sample_accuray = 0.0
                  print("############### %s Total Accuracy = %lf \n" % (self.model_op_name, (total_accuracy / self._op_length)))
            else:
                  for i in range(self._op_length):
                        if files[i] != 'Inference/Blank_WAV_file.wav':
                              vals = session.run(self.running_ops)
                              emotions = ['Anger', 'Happines', 'Sadness', 'Fear', 'Natural']
                              index = np.argmax(vals["predictions"])
                              print(" -------------- Emotion for file %s =  %s \n" % (files[i],emotions[index]))
                              print(" -------------- Raw predictions for file %s =  %s \n" % (files[i], list(map('{:.8f}'.format, vals["predictions_raw"]))))
                              print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>< \n\n")
            
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

class EMO_DB_Config(object):
      dir_name = ['EMO-DB']
      data_set_name = ['EMO-DB']
      train_test_slice = 0.8

class SAVEE_Config(object):
      dir_name = ['SAVEE']
      data_set_name = ['SAVEE']
      train_test_slice = 0.8

class RAVDESS_Config(object):
      dir_name = ['RAVDESS']
      data_set_name = ['RAVDESS']
      train_test_slice = 0.8

class MULTIPLE_DATA_SETS_Config(object):
      dir_name = ['EMO-DB', 'RAVDESS']
      data_set_name = ['EMO-DB', 'RAVDESS']
      train_test_slice = 0.8

class Inference_Config(object):
      dir_name = ['Inference']

#%%
def main():
      ses = tf.Session()

      ser_dp = SER_Data_Producer(EMO_DB_Config())
      ser_dp.import_data(ses)

      train_inputs, train_targets, train_length = ser_dp.train_data
      test_inputs, test_targets, test_length = ser_dp.test_data

      ser_train_model = Speech_Emotion_Recognizer(train_inputs, train_targets, train_length, "Training", True)
      ser_test_model  = Speech_Emotion_Recognizer(test_inputs, test_targets, test_length, "Testing")

      ser_train_model.model()
      ser_test_model.model()

      ser_train_model.create_saver()

      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()

      ser_train_model.initialize_variables(ses)
      epochs = 5
      for epoch in range(epochs):
            print("\n-----------> Epoch %d" % epoch)
            ser_train_model.run_model(ses, writer, merged_summaries)
      ser_train_model.save_model(ses, "/tmp/model.ckpt")
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      ser_test_model.run_model(ses, writer, merged_summaries)

      ses.close()

if __name__ == '__main__':
      main()

#%%
def inference():
      ses = tf.Session()

      if flag_end_to_end:
            ser_dp_inference = Data_Producer_End_to_End_Inference(Inference_Config())
      else:
            ser_dp_inference = Data_Producer_Hand_Crafted_Inference(Inference_Config())

      infr_inputs, inference_length, files = ser_dp_inference.produce_data(ses)

      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()

      ser_inference_model = Speech_Emotion_Recognizer(
      inputs=infr_inputs, op_length=inference_length, model_op_name="Testing", is_training=False, is_inference=True)

      ser_inference_model.create_saver()
      ser_inference_model.restore_model(ses, "/tmp/model.ckpt")

      ser_inference_model.model()
      ser_inference_model.run_model(ses, writer, merged_summaries, files)

      ses.close()

inference()
#%%

