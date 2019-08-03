# %%
import scipy.io.wavfile
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
from python_speech_features import mfcc, logfbank
from tqdm import tqdm
import tensorflow as tf
# %%


class Feature_Extractor(object):
      def __init__(self, directory_name, data_set_name):
            print(
                "------------------ Starting extracting featuers for data set : %s " % data_set_name)
            self.files = self._get_files_from_directory(directory_name)
            self._set_data_set_config(data_set_name)

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

      def _one_hotizize(self, targets):
            """ CONVERT THE LETTERS REPRESENTING EMOTIONS INTO ONE HOT ENCODING
                Arguments:
                targes - list of emotion coressponding to each input file
            """
            targets = [self.e_to_n_mapping[emotion] for emotion in targets]
            return np.eye(self.emotion_number)[targets]

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
            mfcc = librosa.feature.mfcc(
                y=signal, sr=rate, hop_length=260, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            delta_deltas = librosa.feature.delta(delta)
            rms = librosa.feature.rms(
                y=signal, frame_length=640, hop_length=260)
            zcr = librosa.feature.zero_crossing_rate(
                y=signal, frame_length=640, hop_length=260)
            chroma = librosa.feature.chroma_stft(
                y=signal, sr=rate, n_fft=820, win_length=640, hop_length=260)
            rolloff = librosa.feature.spectral_rolloff(
                y=signal, sr=rate, n_fft=820, win_length=640, hop_length=260)

            features = [mfcc, delta, delta_deltas, rms, zcr, chroma, rolloff]
            return features

      def _transform_wave_files(self, files):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                Arguments:
                files - the list of file from which to extract the features
            """
            print("------------------ Extracting audio features from files")
            files = [wav_file for wav_file in files if wav_file[self.emotion_letter_position]
                     in self.e_to_n_mapping.keys()]
            self.features = np.array(
                [self._get_audio_features(wav_file) for wav_file in tqdm(files)])
            self.targets = [wav_file[self.emotion_letter_position]
                            for wav_file in files]
            self.targets = self._one_hotizize(self.targets)

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

      def _reshape_features_for_one_file(self, features):
            """
            """
            steps = features[0].shape[0]
            new_features = np.array(
                [[feature[i] for feature in features] for i in range(steps)])
            return np.array([self._flatten_features(row) for row in new_features])

      def _reshape_features(self, files_features):
            print(
                "------------------ Reshaping features regarding the audio file's frames")
            files_features = np.array([[np.transpose(
                feature) for feature in file_features] for file_features in files_features])
            return np.array([self._reshape_features_for_one_file(file_features) for file_features in tqdm(files_features)])

      def get_featurs_and_targets(self):
            """ THIS FUNCTION WILL BE THE ONE CALLED FROM THE OUTSIDE OF THIS CLASS
                TO OBTAIN THE FEATURES AND TARGETS  
            """
            print("------------------ Processing audio files")
            self._transform_wave_files(self.files)
            self.show_pic(self.features[0])
            self.inputs = self._reshape_features(self.features)
            return self.inputs, self.targets

      def show_pic(self, features):
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

      def set_EMO_DB_config(self):
            self.e_to_n_mapping = {'W': 0, 'F': 1, 'T': 2, 'N': 3}
            self.emotion_number = 4
            self.emotion_letter_position = -6

      def set_SAVEE_config(self):
            self.e_to_n_mapping = {'a': 0, 'h': 1, 's': 2, 'n': 3}
            self.emotion_number = 4
            self.emotion_letter_position = 9

      def set_RAVDESS_config(self):
            self.e_to_n_mapping = {'5': 0, '3': 1, '4': 2, '1': 3}
            self.emotion_number = 4
            self.emotion_letter_position = -17

#%%


class NormalConfig(object):
      dir_name = 'EMO-DB'
      data_set_name = 'EMO-DB'
      train_test_slice = 0.8


class Data_Producer(object):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor(
                config.dir_name, config.data_set_name)
            self._train_test_slice = config.train_test_slice

      def _import_data(self):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR
            """
            self._inputs, self._targets = self._feature_extractor.get_featurs_and_targets()

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
            """ 
            """
            self._import_data()
            self._separate_train_from_test()

            self._train_inputs_dt = tf.data.Dataset.from_generator(
                lambda: self._train_inputs, tf.float64, output_shapes=[None, None])
            self._train_targets_dt = tf.data.Dataset.from_generator(
                lambda: self._train_targets, tf.float64, output_shapes=[None])
            self._test_inputs_dt = tf.data.Dataset.from_generator(
                lambda: self._test_inputs, tf.float64, output_shapes=[None, None])
            self._test_targets_dt = tf.data.Dataset.from_generator(
                lambda: self._test_targets, tf.float64, output_shapes=[None])

            iterator_train_inputs = self._train_inputs_dt.make_one_shot_iterator()
            iterator_train_targets = self._train_targets_dt.make_one_shot_iterator()
            iterator_test_inputs = self._test_inputs_dt.make_one_shot_iterator()
            iterator_test_targets = self._test_targets_dt.make_one_shot_iterator()

            X_train = iterator_train_inputs.get_next()
            y_train = iterator_train_targets.get_next()
            X_test = iterator_train_inputs.get_next()
            y_test = iterator_train_targets.get_next()

            return (X_train, y_train), (X_test, y_test), (self._train_length, self._test_length)


#%%
ses = tf.Session()
dc = Data_Producer(NormalConfig())
(X_train, y_train), (X_test, y_test), (_train_length,
                                       _test_length) = dc.produce_data(ses)

print(ses.run(X_train).shape)
print(ses.run(y_train).shape)

print(ses.run(X_test).shape)
print(ses.run(y_test).shape)

ses.close()
