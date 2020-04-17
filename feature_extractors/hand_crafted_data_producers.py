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

from feature_extractors.hand_crafted_extractor import Feature_Extractor_Hand_Crafted_Training_Testing
from feature_extractors.hand_crafted_extractor import Feature_Extractor_Hand_Crafted_Inference
from util import *

class Data_Producer_Hand_Crafted_Train_Test(object):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_Hand_Crafted_Training_Testing(config.dir_name, config.data_set_name)
            self._train_test_slice = config.train_test_slice
            self._target_domain = config.target_domain[0]

      def _import_data(self, session):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
            """
      def _import_data(self, session):
            self._inputs, self._targets, self._feature_count = self._feature_extractor.get_featurs_and_targets(self._target_domain, session)

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

      def produce_data_train(self, session, name=None):
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

            self._train_inputs_dt = tf.data.Dataset.from_generator(
                lambda: generator_shuffle(self._train_inputs, 1), tf.float32, output_shapes=[None, self._feature_count])
            self._train_targets_dt = tf.data.Dataset.from_generator(
                lambda: generator_shuffle(self._train_targets, 0), tf.float32, output_shapes=[None])

            self._train_inputs_dt = self._train_inputs_dt.repeat()
            self._train_targets_dt = self._train_targets_dt.repeat()

            iterator_train_inputs = self._train_inputs_dt.make_one_shot_iterator()
            iterator_train_targets = self._train_targets_dt.make_one_shot_iterator()

            X_train = iterator_train_inputs.get_next()
            y_train = iterator_train_targets.get_next()

            return (X_train, y_train), self._train_length

      def produce_data_test(self, session, name=None):
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

            self._test_inputs_dt = tf.data.Dataset.from_generator(
                lambda: generator_shuffle(self._test_inputs, 0), tf.float32, output_shapes=[None, self._feature_count])
            self._test_targets_dt = tf.data.Dataset.from_generator(
                lambda: generator_shuffle(self._test_targets, 0), tf.float32, output_shapes=[None])

            self._test_inputs_dt = self._test_inputs_dt.repeat()
            self._test_targets_dt = self._test_targets_dt.repeat()
            
            iterator_test_inputs = self._test_inputs_dt.make_one_shot_iterator()
            iterator_test_targets = self._test_targets_dt.make_one_shot_iterator()

            X_test = iterator_test_inputs.get_next()
            y_test = iterator_test_targets.get_next()

            return (X_test, y_test), self._test_length
#%%
class Data_Producer_Hand_Crafted_Inference(object):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_Hand_Crafted_Inference(config.dir_name)

      def _import_data(self, session):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR
                    -Arguments:
                        session: the tf.Session() the model is running on
            """
            self._features, self._files = self._feature_extractor.get_featurs_and_targets(session)

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
            inference_length = self._features.shape[0]
            feature_count = self._features[0].shape[1]
            self._features_dt = tf.data.Dataset.from_generator(
                lambda: self._features, tf.float32, output_shapes=[None, feature_count]).repeat()

            features = self._features_dt.make_one_shot_iterator()

            inputs = features.get_next()

            return inputs, inference_length, self._files
#%%
