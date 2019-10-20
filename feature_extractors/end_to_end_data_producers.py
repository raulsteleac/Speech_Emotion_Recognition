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

from feature_extractors.end_to_end_extractor import Feature_Extractor_End_to_End_Train_Test
from feature_extractors.end_to_end_extractor import Feature_Extractor_End_to_End_Inference
from util import *

class Data_Producer_End_to_End(object):
      def conv_layer(self, input_data, filter_size, channels_in, channels_out, strides, name="Conv"):
            """  CREATES CONVOLUTIONAL LAYER WITH GIVEN FILTER SIZE AND CHANNELS NUMBERS
            """
            W = tf.get_variable("Weights_Fully_Connected_Layer"+name, dtype=tf.float32, shape=[filter_size, filter_size, channels_in, channels_out])
            return tf.nn.conv2d(input=input_data, filter=W, strides=strides, padding='SAME')

      def _convolutional_feature_extractor(self, stft):
            """ THE FUNCTION BUILDS THE END-TO-END FEATURE EXTRACTION COMPONENT CONSISTING
                OF 2 CONVOLUTIONAL LAYERS COMBINED WITH 2 MAX POOLING LAYERS
                    -Arguments:
                        stft: the framed melspectrograms of the audio files
                    -Returns:
                        conv_out: the features "extracted" after filtering the melspectogram through the convolutional layers   
            """
            self.init = tf.random_normal_initializer(-0.1, 0.1)
            with tf.variable_scope("Convbb", reuse=tf.AUTO_REUSE, initializer=self.init):
                  conv1 = self.conv_layer(input_data=stft, filter_size=8, channels_in=1, channels_out=32, strides=[1, 2, 2, 1], name="conv1")
                  conv2 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding="SAME")
                  conv3 = self.conv_layer(input_data=conv2, filter_size=4, channels_in=32, channels_out=16, strides=[1, 2, 2, 1], name="conv3")
                  conv3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], padding="SAME")
                  conv_out = tf.reshape(conv3, (-1, 1024))
            return conv_out

class Data_Producer_End_to_End_Train_Test(Data_Producer_End_to_End):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_End_to_End_Train_Test(config.dir_name, config.data_set_name)
            self._train_test_slice = config.train_test_slice
            self._target_domain = config.target_domain[0]
            self.seed = 1

      def _import_data(self, session):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
                    -Arguments:
                        session: the tf.Session() the model is running on
            """
            self._inputs, self._targets, self.feature_shape = self._feature_extractor.get_featurs_and_targets(self._target_domain, session)

      def _separate_train_from_test(self):
            """ REGROUP DATA INTO TRAIN DATA AND TEST DATA
                    - given the small number of sample the validation phase is ignored
            """
            self._train_length = int(self._inputs.shape[0]*self._train_test_slice)
            self._test_length = int(self._inputs.shape[0]*(1-self._train_test_slice))

            self._inputs = np.array([inputs.reshape([inputs.shape[0], inputs.shape[1], inputs.shape[2], 1]) for inputs in self._inputs])

            self._train_inputs = self._inputs[0: -self._test_length + 1]
            self._train_targets = self._targets[0: -self._test_length + 1]

            self._test_inputs = self._inputs[-self._test_length:]
            self._test_targets = self._targets[-self._test_length:]

      def produce_data(self, session, name=None):
            """ CONSTRUCTING TF.DATASETS BASED ON THE FEATURES EXTRACTED
                    -Arguments:
                        session: the tf.Session() the model is running on
                    -Returns:
                        (X_train, y_train) - pair representing one instance of the train data
                        (X_test, y_test) - pair representing one instance of the train data
                        (self._train_length, self._test_length) - pair representing the length of the train and test data                
            """
            self._import_data(session)
            self._separate_train_from_test()

            self._train_inputs_dt = tf.data.Dataset.from_generator(lambda: generator_shuffle(self._train_inputs, 1), tf.float32, output_shapes=self.feature_shape)
            self._train_targets_dt = tf.data.Dataset.from_generator(lambda: generator_shuffle(self._train_targets, 0), tf.float32, output_shapes=[None])
            self._test_inputs_dt = tf.data.Dataset.from_generator(lambda: generator_shuffle(self._test_inputs, 0), tf.float32, output_shapes=self.feature_shape)
            self._test_targets_dt = tf.data.Dataset.from_generator(lambda: generator_shuffle(self._test_targets, 0), tf.float32, output_shapes=[None])

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

      def _import_data(self, session):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
                    -Arguments:
                        session: the tf.Session() the model is running on
            """
            self._features, self._files = self._feature_extractor.get_features_and_files(session)

      def produce_data(self, session, name=None):
            """ CONSTRUCTING TF.DATASETS BASED ON THE FEATURES EXTRACTED
                    -Arguments:
                        session: the tf.Session() the model is running on
                    -Returns:
                        inputs - the features extracted from the convolutional layers
                        inference_length - the number of files in the inference folder
                        self._files - the names of the files in the inference folder to pretty print              
            """
            self._import_data(session)
            self._features = np.array([_inputs.reshape([_inputs.shape[0], _inputs.shape[1], _inputs.shape[2], 1]) for _inputs in self._features])

            inference_length = self._features.shape[0]
            self._features_dt = tf.data.Dataset.from_generator(lambda: self._features, tf.float32, output_shapes=[None, self._features[0].shape[1], self._features[0].shape[2], 1]).repeat()
            features = self._features_dt.make_one_shot_iterator()
            inputs = features.get_next()
            inputs = self._convolutional_feature_extractor(inputs)

            return inputs, inference_length, self._files


#%%
