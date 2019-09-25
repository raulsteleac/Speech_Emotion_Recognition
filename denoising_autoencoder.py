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

class DAE(object):
      def __init__(self, fit_inputs, hidden_layer_dimension):
            self._fit_inputs = fit_inputs
            self._hidden_layer_dimension = hidden_layer_dimension
            self.init = tf.random_normal_initializer(-0.1, 0.1)
      
      def autoencoder_model(self):
            shape = [None, *self._fit_inputs[0].shape[1:]]
            with tf.variable_scope("Denoising_autoencoder_layer", reuse=tf.AUTO_REUSE, initializer=self.init):
                  self.autoencoder_input = tf.placeholder(tf.float64, shape)
                  autoencoder_input_noisy = self.autoencoder_input + 1.0 * tf.random_normal(tf.shape(self.autoencoder_input), dtype=tf.float64)
                  self.encoder_layer_1 = tf.layers.dense(autoencoder_input_noisy,  self._hidden_layer_dimension, activation=tf.nn.relu, name="First_Encoder_Layer")
                  self.encoder_output = tf.layers.dense(self.encoder_layer_1,self._hidden_layer_dimension, activation=tf.nn.relu, name="Second_Encoder_Layer")

                  self.decoder_layer_1 = tf.layers.dense(self.encoder_output, self._hidden_layer_dimension, activation=tf.nn.relu, name="First_Decoder_Layer")
                  self.decoder_output = tf.layers.dense(self.decoder_layer_1, self.autoencoder_input.shape[1], name="Output_Decoder_Layer" )

                  reconstruction_loss = tf.reduce_mean(tf.square(self.decoder_output - self.autoencoder_input))
                  self.autoencoder_optimizer = tf.train.AdamOptimizer(0.001).minimize(reconstruction_loss)

      def autoencoder_fit(self, epochs, session):
            print("---------- Autoencoder fitting")
            session.run(tf.global_variables_initializer())
            for _ in range(epochs):
                  for frame_inputs in self._fit_inputs:
                        session.run(self.autoencoder_optimizer, feed_dict={self.autoencoder_input: frame_inputs})

      def autoencoder_transform(self, data, session):
            return np.array([session.run(self.decoder_output, feed_dict={self.autoencoder_input: frame_inputs}) for frame_inputs in data])

      def create_saver(self):
            self.saver = tf.train.Saver()

      def save_model(self, ses, path):
            self.saver.save(ses, path)

      def restore_model(self, ses, path):
            self.saver.restore(ses, path)


      
