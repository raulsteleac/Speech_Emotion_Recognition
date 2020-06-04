# %%
import os
try:
    os.chdir('/home/raulslab/work/Speech_Emotion_Recognition')
    print(os.getcwd())
except:
      print("Can't change the Current Working Directory")
      pass

import librosa
import numpy as np
import tensorflow as tf
from util import *

from tqdm import tqdm
# %%
class Feature_Extractor(object):
      _dae = None
      _target_domain = None

      def __init__(self, directory_name_list, thread):
            """  FETCH THE .WAV FILES FROM ALL THE DATASET DIRECTORIES ORIGINATING FROM THIS ONE
                  -Arguments:
                        directory_name_list : list of directories containing the datasets
            """
            self.thread = thread
            print("------------------ Starting extracting featuers for data set : %s " % ' '.join(directory_name_list))
            self.files = [get_files_from_directory(directory_name) for directory_name in directory_name_list]
            self.thread = thread

      def _set_data_set_config(self, data_set_name):
            """ SET CONFIGURATION DEPENDING ON THE DATA SET GIVEN BY THE ARGUMENT OF THIS FUNCTION
                  -Arguments:
                        data_set_name - name of the currently used data set e.g. : EMO-DB, SAVEE
            """
            if data_set_name == 'EMO-DB':
                  self.set_EMO_DB_config()

            if data_set_name == 'SAVEE':
                  self.set_SAVEE_config()

            if data_set_name == 'RAVDESS':
                  self.set_RAVDESS_config()

            if data_set_name == 'ENTERFACE':
                  self.set_ENTERFACE_Config()

            if data_set_name == 'EMOVO':
                  self.set_EMOVO_Config()

            if data_set_name == 'MAV':
                  self.set_MAV_Config()

            if data_set_name == 'MELD':
                  self.set_MELD_Config()

            if data_set_name == 'JL':
                  self.set_JL_Config()

            if data_set_name == 'InrP':
                  self.set_InrP_Config()

      def set_EMO_DB_config(self):
            self.e_to_n_mapping = {'W': 0, 'F': 1, 'T': 2, 'N': 3}#, 'E': 4, 'A': 5, 'L': 6} 
            self.emotion_number = 4
            self.emotion_letter_position = -6
            self.hz = 16000

      def set_SAVEE_config(self):
            self.e_to_n_mapping = {'a': 0, 'h': 1, 's': 2, 'n': 3}#, 'd':4, 'f': 5, 'u':6} 
            self.emotion_number = 4
            self.emotion_letter_position = 19
            self.hz = 44100

      def set_RAVDESS_config(self):
            self.e_to_n_mapping = {'5': 0, '3': 1, '4': 2, '1': 3}#, '7':4, '6':5, '2':6, '8':7} 
            self.emotion_number = 4
            self.emotion_letter_position = -17
            self.hz = 48000

      def set_ENTERFACE_Config(self):
            self.e_to_n_mapping = {'a': 0, 'h': 1, 's': 2, 'n': 3}#'d':4, 'f': 5} 
            self.emotion_number = 4
            self.emotion_letter_position = -8
            self.hz = 44100

      def set_EMOVO_Config(self):
            self.e_to_n_mapping = {'r': 0, 'g': 1, 't': 2, 'n': 3}#, 'd':4, 'p': 5, 's': 6} 
            self.emotion_number = 4
            self.emotion_letter_position = -13
            self.hz = 48000

      def set_MAV_Config(self):
            self.e_to_n_mapping = {'a': 0, 'h': 1, 's': 2, 'n': 3}#, 'd':4, 'f': 5} 
            self.emotion_number = 4
            self.emotion_letter_position = 38
            self.hz = 44100

      def set_MELD_Config(self):
            self.e_to_n_mapping = {'a': 0, 'j': 1, 's': 2,'n': 3}
            self.emotion_number = 4
            self.emotion_letter_position = 26
            self.hz = 16000

      def set_JL_Config(self):
            self.e_to_n_mapping = {'a': 0, 'h': 1, 's': 2, 'n': 3}
            self.emotion_number = 4
            self.emotion_letter_position = 16
            self.hz = 44100

      def set_InrP_Config(self):
            self.e_to_n_mapping = {'A': 0, 'H': 1, 'S': 2, 'N': 3}  # , 'D': 4, 'F':5}
            self.emotion_number = 4
            self.emotion_letter_position = -5
            self.hz = 48000

      def show_pic(self, feature):
            pass
      
      def _get_audio_features(self, wav_file):
            pass

class Feature_Extractor_End_to_End(Feature_Extractor):
      def __init__(self, directory_name_list, thread):
            super().__init__(directory_name_list, thread)
            self.feature_names = ['STFT']

      def reshape_frames(self, stft, window_length):
            """  RESHAPE THE SPECTOGRAM INTO WINDOWS OF SHAPE 128x128 
                  -Arguments:
                        stft : The spectogram of the audio signal
                  -Returns:
                        The reshaped signal that will be passed to the convolutional layer
            """
            # if self.thread != None:
            #       self.thread.print_stats.emit("------------------ Reshaping features regarding the audio file's frames")
            stft = np.transpose(stft)
            window_nr = (stft.shape[0] // window_length + 1) * window_length
            pad_size = window_nr - stft.shape[0]
            stft = np.pad(stft, ((0, pad_size), (0, 1)), 'edge')
            conv_frames = np.array(([stft[i * window_length:(i+1) * window_length]
                                     for i in range(int(stft.shape[0]/(stft.shape[1]) + 1))]))
            return conv_frames[:, :, 0:window_length]

      def _get_audio_features(self, wav_file):
            """ EXTRACT THE AUDIO FEATURES FROM THE .WAV FILES USING THE LIBROSA LIBRARY
                  -Arguments:
                        wav_file - the name of the .wav file from which to extract the features
                  -Local variables:
                        stft - the coefficients of the Mel Frequency Cepstral for each frame
                  -Returns:
                        stft - -//-
            """
            signal, _ = librosa.load(wav_file, sr=16000)
            librosa.core.time_to_frames
            stft = librosa.feature.melspectrogram(signal, n_fft=512, win_length=128, hop_length=32, center=False)
            return stft

class Feature_Extractor_Hand_Crafted(Feature_Extractor):
      def __init__(self, directory_name_list, thread):
            super().__init__(directory_name_list, thread)
            self.feature_names = ['MFCC', 'Delta', 'Delta-Deltas', 'RMS', 'ZCR', 'Chrmoa', 'Roll-off']

      def _flatten_features(self, row):
            """  FLATTENS THE ROWS THAT REPRESENT THE FEATURES EXTRACTED FROM THE AUDIO FILES PER FRAME
                  -Arguments:
                        row : the raw shaped features extracted from the audio files for one frame
                  -Returns:
                        new_features : the features flattened in shape (, 75)
            """
            new_features = np.array([])
            for feature in row:
                  new_values = np.array([])
                  for val in feature:
                        new_values = np.append(new_values, values=val)
                  new_features = np.append(new_features, new_values)
            return new_features

      def _reshape_features_for_one_file(self, features):
            """  APPLY THE FLATTEN FUNCTION ON EACH ROW OF THE FEATURE MATRIX
                  -Arguments:
                        features : the feature matrix to be resahped
                  -Returns:
                        Numpy array representing the newly transofrmed feature matrix
            """
            steps = features[0].shape[0]
            new_features = np.array([[feature[i] for feature in features] for i in range(steps)])
            return np.array([self._flatten_features(row) for row in new_features])

      def _reshape_features(self, files_features):
            """ FIRST DOES A TRANSPOSE TO OBTAIN INSTANCES IN THE FORM (N, 75) from (75, N) 
                THEN CALLS THE RESHAPING FUNCTION FOR THE FEATURE SET OF EACH FILE
                  -Arguments:
                        
                  -Returns:
                        The reshaped version of the initial data-set whose every instance is of the form
                        (N, 75), where N represents the number of frames, differing for each audio file.  
            """
            # print("------------------ Reshaping features regarding the audio file's frames")
            # if self.thread != None:
            #       self.thread.print_stats.emit("------------------ Reshaping features regarding the audio file's frames")
            files_features = np.array([[np.transpose(feature) for feature in file_features] for file_features in files_features])
            return np.array([self._reshape_features_for_one_file(file_features) for file_features in tqdm(files_features)])

      def _get_audio_features(self, wav_file):
            """ EXTRACT THE AUDIO FEATURES FROM THE .WAV FILES USING THE LIBROSA LIBRARY
                  -Arguments:
                        wav_file - the name of the .wav file from which to extract the features
                        Local variables:
                        mfcc - the coefficients of the Mel Frequency Cepstral for each frame
                        delta - the deltas of the mfcc (analogue to speed) for each frame
                        delta_deltas - the deltas of the delta (analogue to acceleration) for each frame
                        rms - the root mean square of the amplitude of the signal in each frame
                        zcr - Zero crossing rate, the rate at which a signal changes its sign during one frame
                        chroma - measures of the different pitch classes, 12 for every frame
                        rollof -  measures of the frequency that falls under some percentage (cutoff) of the total energy of the spectrum
                  -Returns:
                        features: list of all the hand-crafted features extracted from the audio file 
            """
            signal, rate = librosa.load(wav_file, 16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=rate, hop_length=400, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            delta_deltas = librosa.feature.delta(delta)
            rms = librosa.feature.rms(y=signal, frame_length=1020, hop_length=400)
            zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=1020, hop_length=400)
            chroma = librosa.feature.chroma_stft(y=signal, sr=rate, n_fft=1140, win_length=1020, hop_length=400)
            rolloff = librosa.feature.spectral_rolloff(y=signal, sr=rate, n_fft=1140, win_length=1020, hop_length=400)

            features = [mfcc, delta, delta_deltas, rms, zcr, chroma, rolloff]
            return features 
            # return np.reshape(np.transpose(features),(-1,60))

#%%
