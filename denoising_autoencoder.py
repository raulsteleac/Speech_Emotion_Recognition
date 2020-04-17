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
from tqdm import tqdm

class DAE(object):
      def __init__(self, hidden_layer_dimension):
            self._hidden_layer_dimension = hidden_layer_dimension
            self.init = tf.random_normal_initializer(-0.1, 0.1)

      def conv_layer(self, input_data, filter_size, channels_in, channels_out, strides, conv_layer_dropout, name="Conv"):
            """  CREATES CONVOLUTIONAL LAYER WITH GIVEN FILTER SIZE AND CHANNELS NUMBERS
            """
            W = tf.get_variable("Weights_Fully_Connected_Layer"+name, dtype=tf.float32, shape=[filter_size, filter_size, channels_in, channels_out])
            return (tf.nn.dropout(tf.nn.conv2d(input=input_data, filter=W, strides=strides, padding='SAME', use_cudnn_on_gpu=True), conv_layer_dropout))
      
      def autoencoder_model(self, inputs):
            with tf.variable_scope("Denoising_autoencoder_layer", reuse=tf.AUTO_REUSE, initializer=self.init):
                  autoencoder_input_noisy = inputs + 0.1 * tf.random_normal(tf.shape(inputs), dtype=tf.float32)
                  self.encoder_output = tf.layers.dense(autoencoder_input_noisy, self._hidden_layer_dimension, activation=tf.nn.tanh, name="Encoder_Layer")
                  self.decoder_output = tf.layers.dense(self.encoder_output, inputs.shape[1], name="Output_Decoder_Layer")

                  reconstruction_loss = tf.reduce_sum(tf.square(self.decoder_output - inputs))
                  self.autoencoder_optimizer = tf.train.AdamOptimizer(0.00001).minimize(reconstruction_loss)

      def autoencoder_fit_unsupervised(self, epochs, inputs, session):
            print("---------- Autoencoder fitting")
            shape = [None, *inputs[0].shape[1:]]
            autoencoder_input_ = tf.placeholder(tf.float32, shape)
            self.autoencoder_model(autoencoder_input_)
            session.run(tf.global_variables_initializer())
            for _ in tqdm(range(epochs)):
                  for frame_inputs in inputs:
                        session.run(self.autoencoder_optimizer, feed_dict={autoencoder_input_: frame_inputs})

      def autoencoder_fit_supervised(self, inputs):
            self.autoencoder_model(inputs)
            return self.decoder_output

      def autoencoder_transform(self, inputs, ses): 
            shape = [None, *inputs[0].shape[1:]]
            autoencoder_input_ = tf.placeholder(tf.float32, shape)
            self.autoencoder_model(autoencoder_input_)
            outputs = np.array([ses.run(self.decoder_output, feed_dict={autoencoder_input_: frame_inputs}) for frame_inputs in inputs])
            return outputs

      def create_saver(self):
            self.saver = tf.train.Saver()

      def save_model(self, ses, path):
            self.saver.save(ses, path)

      def restore_model(self, ses, path):
            self.saver.restore(ses, path)


      
