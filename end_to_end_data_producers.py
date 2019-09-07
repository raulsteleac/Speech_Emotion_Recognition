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

from end_to_end_extractor import Feature_Extractor_End_to_End_Train_Test
from end_to_end_extractor import Feature_Extractor_End_to_End_Inference

class Data_Producer_End_to_End(object):
      def conv_layer(self, input_data, filter_size, channels_in, channels_out, strides, name="Conv"):
            W = tf.get_variable("Weights_Fully_Connected_Layer"+name, dtype=tf.float32,
                                shape=[filter_size, filter_size, channels_in, channels_out])
            b = tf.get_variable("Biases_Fully_Connected_Layer" +
                                name, dtype=tf.float32, shape=[channels_out])
            return tf.nn.conv2d(input=input_data, filter=W, strides=strides, padding='SAME') + b

      def _convolutional_feature_extractor(self, stft):
        self.init = tf.random_normal_initializer(-0.1, 0.1)
        with tf.variable_scope("Convbb", reuse=tf.AUTO_REUSE, initializer=self.init):
            #stft = tf.reshape(stft, (stft.shape[0], stft[0].shape[0],  stft[0].shape[1], 1))
            conv1 = self.conv_layer(input_data=stft, filter_size=128, channels_in=1 , channels_out=32, strides=[1, 2, 2, 1], name="conv1")
            conv2 = self.conv_layer(input_data=conv1, filter_size=64, channels_in=32, channels_out=32, strides=[1, 2, 2, 1], name="conv2")
            conv3 = self.conv_layer(input_data=conv2, filter_size=64, channels_in=32, channels_out=32, strides=[1, 1, 1, 1], name="conv3")
            conv4 = self.conv_layer(input_data=conv3, filter_size=32, channels_in=32, channels_out=32, strides=[1, 8, 4, 1], name="conv4")
            conv_out = tf.reshape(conv4, (-1, 1024))
            outputs = tf.layers.dense(
                conv_out, 1024, activation=tf.nn.relu, name="Forth_Fully_Connected_Layer")
        return outputs

class Data_Producer_End_to_End_Train_Test(Data_Producer_End_to_End):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_End_to_End_Train_Test(config.dir_name, config.data_set_name)
            self._train_test_slice = config.train_test_slice

      def _import_data(self):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
            """
            self._inputs, self._targets, self.feature_shape = self._feature_extractor.get_featurs_and_targets()

      def _separate_train_from_test(self):
            """ REGROUP DATA INTO TRAIN DATA AND TEST DATA
                - given the small number of sample the validation phase is ignored
            """
            self._train_length = int(
                self._inputs.shape[0]*self._train_test_slice)
            self._test_length = int(
                self._inputs.shape[0]*(1-self._train_test_slice))

            self._train_inputs = self._inputs[0: -self._test_length + 1]
            self._train_targets = self._targets[0: -self._test_length + 1]

            self._test_inputs = self._inputs[-self._test_length:]
            self._test_targets = self._targets[-self._test_length:]

      def produce_data(self, session, name=None):
            """ CONSTRUCTING TF.DATASETS BASED ON THE FEATURES EXTRACTED
                Returns:
                (X_train, y_train) - pair representing one instance of the train data
                (X_test, y_test) - pair representing one instance of the train data
                (self._train_length, self._test_length) - pair representing the length of the train and test data                
            """
            self._import_data()
            self._separate_train_from_test()

            self._train_inputs_dt = tf.data.Dataset.from_generator(
                lambda: self._train_inputs, tf.float32, output_shapes=self.feature_shape)
            self._train_targets_dt = tf.data.Dataset.from_generator(
                lambda: self._train_targets, tf.float32, output_shapes=[None])
            self._test_inputs_dt = tf.data.Dataset.from_generator(
                lambda: self._test_inputs, tf.float32, output_shapes=self.feature_shape)
            self._test_targets_dt = tf.data.Dataset.from_generator(
                lambda: self._test_targets, tf.float32, output_shapes=[None])

            self._train_inputs_dt = self._train_inputs_dt.repeat()
            self._train_targets_dt = self._train_targets_dt.repeat()
            self._test_inputs_dt = self._test_inputs_dt.repeat()
            self._test_targets_dt = self._test_targets_dt.repeat()

            iterator_train_inputs = self._train_inputs_dt.make_one_shot_iterator()
            iterator_train_targets = self._train_targets_dt.make_one_shot_iterator()
            iterator_test_inputs = self._test_inputs_dt.make_one_shot_iterator()
            iterator_test_targets = self._test_targets_dt.make_one_shot_iterator()

            X_train = iterator_train_inputs.get_next()
            y_train = iterator_train_targets.get_next()
            X_test = iterator_test_inputs.get_next()
            y_test = iterator_test_targets.get_next()

            X_train = self._convolutional_feature_extractor(X_train)
            X_test = self._convolutional_feature_extractor(X_test)
            session.run(tf.global_variables_initializer())

            return (X_train, y_train), (X_test, y_test), (self._train_length, self._test_length)
#%%
class Data_Producer_End_to_End_Inference(Data_Producer_End_to_End):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_End_to_End_Inference(config.dir_name)

      def _import_data(self):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
            """
            self._features, self._files = self._feature_extractor.get_features_and_files()

      def produce_data(self, session, name=None):
            """ CONSTRUCTING TF.DATASETS BASED ON THE FEATURES EXTRACTED
                Returns:
                (X_train, y_train) - pair representing one instance of the train data
                (X_test, y_test) - pair representing one instance of the train data
                (self._train_length, self._test_length) - pair representing the length of the train and test data                
            """
            self._import_data()

            inference_length = self._features.shape[0]
            self._features_dt = tf.data.Dataset.from_generator(lambda: self._features, tf.float32, output_shapes=[None, self._features[0].shape[1], self._features[0].shape[2], 1]).repeat()
            features = self._features_dt.make_one_shot_iterator()
            inputs = features.get_next()
            inputs = self._convolutional_feature_extractor(inputs)
            session.run(tf.global_variables_initializer())

            return inputs, inference_length, self._files
