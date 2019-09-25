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
# %%
class Feature_Extractor(object):
      _dae = None
      _target_domain = None

      def __init__(self, directory_name_list):
            print(
                "------------------ Starting extracting featuers for data set : %s " % ' '.join(directory_name_list))
            self.files = [self._get_files_from_directory(directory_name) for directory_name in directory_name_list]

      def _get_files_from_directory(self, dir_name):
            """ RETURN A LIST OF THE FILES IN THE ARGUMENT OF THIS FUNCTION
                Arguments:
                dir_name - path to the directory where the *.wav files are stored
            """
            list_of_files = np.array([])
            for root, dirs, files in os.walk(dir_name):
                    for f in files:
                        if f.endswith(".wav"):
                                list_of_files = np.append(
                                    list_of_files, os.path.join(root, f))
            return list_of_files

class Feature_Extractor_End_to_End(Feature_Extractor):
      def __init__(self, directory_name_list):
            super().__init__(directory_name_list)

      def reshape_framse(self, stft):
            window_length = 128
            stft = np.transpose(stft)
            window_nr = (stft.shape[0] // window_length + 1) * window_length
            pad_size = window_nr - stft.shape[0]
            stft = np.pad(stft, ((0, pad_size), (0, 0)), 'edge')
            conv_frames = np.array([stft[i * 128:(i+1) * 128]
                                    for i in range(int(stft.shape[0]/(stft.shape[1])+1))])
            return conv_frames[:, :, 0:128]

      def show_pic(self, feature):
            """ LOG FUNCTION THAT PRINTS A PLOT FOR EACH FEATURE (MFCC, ZCR,.etc) OF ONE FILE
            """
            i = 1
            names = ['STFT']
            plt.figure(figsize=(60, 20))
            plt.subplot(1, 1, i)
            plt.title(names[i-1])
            librosa.display.specshow(feature)
            plt.colorbar()
            i = i+1
            plt.show()
      
      def _get_audio_features(self, wav_file):
            """ EXTRACT THE AUDIO FEATURES FROM THE .WAV FILES USING THE LIBROSA LIBRARY
                Arguments:
                wav_file - the name of the .wav file from which to extract the features
                Local variables:
                stft - the coefficients of the Mel Frequency Cepstral for each frame
            """
            signal, rate = librosa.load(wav_file, 16000)
            librosa.core.time_to_frames
            stft = np.abs(librosa.stft(signal, n_fft=256,win_length=128, hop_length=32, center=False))
            return stft

class Feature_Extractor_Hand_Crafted(Feature_Extractor):
      def __init__(self, directory_name_list):
            super().__init__(directory_name_list)

      def _reshape_features_for_one_file(self, features):
            """
            """
            steps = features[0].shape[0]
            new_features = np.array([[feature[i] for feature in features] for i in range(steps)])
            return np.array([self._flatten_features(row) for row in new_features])

      def _reshape_features(self, files_features):
            """ FIRST DOES A TRANSPOSE TO OBTAIN INSTANCES IN THE FORM (N, 75) from (75, N) 
                THEN CALLS THE RESHAPING FUNCTION FOR THE FEATURE SET OF EACH FILE
                Return:
                The reshaped version of the initial data-set whose every instance is of the form
                (N, 75), where N represents the number of frames, differing for each audio file.  
            """
            print("------------------ Reshaping features regarding the audio file's frames")
            files_features = np.array([[np.transpose(feature) for feature in file_features] for file_features in files_features])
            return np.array([self._reshape_features_for_one_file(file_features) for file_features in tqdm(files_features)])

      def show_pic(self, features):
            """ LOG FUNCTION THAT PRINTS A PLOT FOR EACH FEATURE (MFCC, ZCR,.etc) OF ONE FILE
            """
            i = 1
            names = ['MFCC', 'Delta', 'Delta-Deltas',
                     'RMS', 'ZCR', 'Chrmoa', 'Roll-off']
            plt.figure(figsize=(30, 20))
            for signal in features:
                  plt.subplot(7, 2, i)
                  plt.title(names[i-1])
                  librosa.display.specshow(signal)
                  plt.colorbar()
                  i = i+1
            plt.show()

      def _get_audio_features(self, wav_file):
            """ EXTRACT THE AUDIO FEATURES FROM THE .WAV FILES USING THE LIBROSA LIBRARY
                Arguments:
                wav_file - the name of the .wav file from which to extract the features
                Local variables:
                mfcc - the coefficients of the Mel Frequency Cepstral for each frame
                delta - the deltas of the mfcc (analogue to speed) for each frame
                delta_deltas - the deltas of the delta (analogue to acceleration) for each frame
                rms - the root mean square of the amplitude of the signal in each frame
                zcr - Zero crossing rate, the rate at which a signal changes its sign during one frame
                chroma - measures of the different pitch classes, 12 for every frame
                rollof -  measures of the frequency that falls under some percentage (cutoff) of the total energy of the spectrum
            """
            signal, rate = librosa.load(wav_file, 16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=rate, hop_length=260, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            delta_deltas = librosa.feature.delta(delta)
            rms = librosa.feature.rms(y=signal, frame_length=640, hop_length=260)
            zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=640, hop_length=260)
            chroma = librosa.feature.chroma_stft(y=signal, sr=rate, n_fft=820, win_length=640, hop_length=260)
            rolloff = librosa.feature.spectral_rolloff(y=signal, sr=rate, n_fft=820, win_length=640, hop_length=260)

            features = [mfcc, delta, delta_deltas, rms, zcr, chroma, rolloff]
            return features

      def _flatten_features(self, row):
            """ FLATTENS THE MATRICIES, OF DIFERENT SHAPE, THAT REPRESENT THE FEATURES EXTRACTED FROM THE AUDIO FILES
            """
            new_features = np.array([])
            for feature in row:
                  new_values = np.array([])
                  for val in feature:
                        new_values = np.append(new_values, values=val)
                  new_features = np.append(new_features, new_values)
            return new_features
