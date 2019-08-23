# %%
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'Speech_Emotion_Recognition'))
    print(os.getcwd())
except:
    pass

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from feature_extractors import Feature_Extractor_Hand_Crafted

class Feature_Extractor_Hand_Crafted_Training_Testing(Feature_Extractor_Hand_Crafted):
      def __init__(self, directory_name_list, data_set_name_list):
            super().__init__(directory_name_list)
            self._data_set_name_list = data_set_name_list

      def _set_data_set_config(self, data_set_name):
            """ SET CONFIGURATION DEPENDING ON THE DATA SET GIVEN BY THE ARGUMENT OF THIS FUNCTION
                Arguments:
                data_set_name - name of the currently used data set e.g. : EMO-DB, SAVEE
            """
            if data_set_name == 'EMO-DB':
                  self.set_EMO_DB_config()

            if data_set_name == 'RAVDESS':
                  self.set_RAVDESS_config()

            if data_set_name == 'SAVEE':
                  self.set_SAVEE_config()

      def _one_hotizize(self, targets):
            """ CONVERT THE LETTERS REPRESENTING EMOTIONS INTO ONE HOT ENCODING
                Arguments:
                targes - list of emotion coressponding to each input file
                Returns:
                The one-hot encoded version of the targets
            """
            targets = [self.e_to_n_mapping[emotion] for emotion in targets]
            return np.eye(self.emotion_number)[targets]

      def _transform_wave_files(self, files):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                Arguments:
                files - the list of file from which to extract the features
            """
            print("------------------ Extracting audio features from files")
            files = [wav_file for wav_file in files if wav_file[self.emotion_letter_position]
                     in self.e_to_n_mapping.keys()]
            print(files[0])
            self.features = np.array([self._get_audio_features(wav_file) for wav_file in tqdm(files)])
            targets = [wav_file[self.emotion_letter_position]for wav_file in files]
            self.targets = np.append(self.targets, self._one_hotizize(targets))

      def _shuffle_data(self, inputs, targets):
            """ SHUFFLE BOTH THE INPUTS AND THE TARGETS IN THE SAME MANNER
                Returns:
                inputs, targets - the shuffled version of the data  
            """
            shuffle = np.arange(inputs.shape[0])
            np.random.seed(17)
            np.random.shuffle(shuffle)
            inputs = inputs[shuffle]
            targets = targets[shuffle]
            return inputs, targets

      def get_featurs_and_targets(self):
            """ THIS FUNCTION WILL BE THE ONE CALLED FROM THE OUTSIDE OF THIS CLASS
                TO OBTAIN THE FEATURES AND TARGETS 
                Returns:
                self.inputs, self.targets - represents the list of features and targets propagated outside this class 
            """
            print("------------------ Processing audio files")
            self.inputs = np.array([])
            self.targets = np.array([[]])

            for files, ds_name in zip(self.files, self._data_set_name_list):
                self._set_data_set_config(ds_name)
                self._transform_wave_files(files)
                self.show_pic(self.features[0])
                self.inputs = np.append(self.inputs, self._reshape_features(self.features))

            self.targets = np.reshape(self.targets, (-1, self.emotion_number))
            self.inputs, self.targets = self._shuffle_data(
                self.inputs, self.targets)
            return self.inputs, self.targets, self.inputs[0].shape[1]

      def set_EMO_DB_config(self):
            self.e_to_n_mapping = {'W': 0, 'F': 1, 'T': 2, 'A': 3, 'N': 4}
            self.emotion_number = 5
            self.emotion_letter_position = -6

      def set_SAVEE_config(self):
            self.e_to_n_mapping = {'a': 0, 'h': 1, 's': 2, 'f': 3, 'n': 4}
            self.emotion_number = 5
            self.emotion_letter_position = 9

      def set_RAVDESS_config(self):
            self.e_to_n_mapping = {'5': 0, '3': 1, '4': 2, '6': 3, '1': 4}
            self.emotion_number = 5
            self.emotion_letter_position = -17


class Feature_Extractor_Hand_Crafted_Inference(Feature_Extractor_Hand_Crafted):
      def __init__(self, directory_name_list):
            super().__init__(directory_name_list)

      def get_features_and_files(self):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                Arguments:
                files - the list of file from which to extract the features
            """
            print("------------------ Extracting audio features from files")
            self.files = self.files[0]
            print("List of files is : %s" % self.files)
            self.features = np.array([self._get_audio_features(wav_file) for wav_file in tqdm(self.files)])
            self.show_pic(self.features[0])
            self.features = self._reshape_features(self.features)
            print("------------------------------------------------------------------------")
            return self.features, self.files
