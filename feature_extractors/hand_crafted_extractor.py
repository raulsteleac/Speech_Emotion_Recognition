# %%
import os
import librosa
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from feature_extractors.feature_extractor import Feature_Extractor
from feature_extractors.feature_extractor import Feature_Extractor_Hand_Crafted
from util import *

class Feature_Extractor_Hand_Crafted_Training_Testing(Feature_Extractor_Hand_Crafted):
      def __init__(self, directory_name_list, data_set_name_list, thread=None):
            super().__init__(directory_name_list, thread)
            self._data_set_name_list = data_set_name_list

      def _transform_wave_files(self, files):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                    -Arguments:
                        files - the list of file from which to extract the features
            """
            print("------------------ Extracting audio features from files")
            files = [wav_file for wav_file in files if wav_file[self.emotion_letter_position]
                     in self.e_to_n_mapping.keys()]

            self.features = np.array([self._get_audio_features(wav_file) for wav_file in tqdm(files)])
            targets = [wav_file[self.emotion_letter_position]for wav_file in files]
            self.targets = np.append(self.targets,  one_hotizize(targets, self.e_to_n_mapping, self.emotion_number))

      def get_featurs_and_targets(self, session):
            """ THIS FUNCTION WILL BE THE ONE CALLED FROM THE OUTSIDE OF THIS CLASS
                TO OBTAIN THE FEATURES AND TARGETS 
                    -Arguments:
                        targer_domain: the chosen dataset on witch DAE trains
                        session: the tf.Session() the model is running on
                    -Returns:
                        self.inputs, self.targets - represents the list of features and targets propagated outside this class 
            """
            print("------------------ Processing audio files")
            if self.thread != None:
                  self.thread.print_stats.emit("------------------ Processing audio files")
            self.inputs = np.array([])
            self.targets = np.array([[]])

            for files, ds_name in zip(self.files, self._data_set_name_list):
                if self.thread != None:
                    self.thread.print_stats.emit("------------------ Extracting audio features from %s " % ds_name)
                self._set_data_set_config(ds_name)
                self._transform_wave_files(files)
                self.features = self._reshape_features(self.features)
                self.inputs = np.append(self.inputs, self.features)
            self.targets = np.reshape(self.targets, (-1, self.emotion_number))
            self.inputs, self.targets = shuffle_data(self.inputs, self.targets)
            return self.inputs, self.targets, self.inputs[0].shape[1]

class Feature_Extractor_Hand_Crafted_Inference(Feature_Extractor_Hand_Crafted):
      def __init__(self, directory_name_list, thread):
            super().__init__(directory_name_list, thread)

      def get_featurs_and_targets(self, session):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                    -Arguments:
                        session: the tf.Session() the model is running on
                    -Returns:
                        self.features - represents the list of features propagated outside this class 
                        self.files - the list of files from wich the features were extracted from
            """
            print("------------------ Extracting audio features from files")
            self.files = self.files[0]
            print("List of files is : %s" % self.files)
            self.features = np.array([self._get_audio_features(wav_file) for wav_file in tqdm(self.files)])
            self.features = self._reshape_features(self.features)
            print(self.features[0].shape)
            print("------------------------------------------------------------------------")
            return self.features, self.files
