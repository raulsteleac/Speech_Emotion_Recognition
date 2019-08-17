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
            new_features = np.array([[feature[i] for feature in features] for i in range(steps)])
            return np.array([self._flatten_features(row) for row in new_features])

      def _reshape_features(self, files_features):
            """ FIRST DOES A TRANSPOSE TO OBTAIN INSTANCES IN THE FORM (N, 75) from (75, N) 
                THEN CALLS THE RESHAPING FUNCTION FOR THE FEATURE SET OF EACH FILE
                Return:
                The reshaped version of the initial data-set whose every instance is of the form
                (N, 75), where N represents the number of frames, differing for each audio file.  
            """
            print(
                "------------------ Reshaping features regarding the audio file's frames")
            files_features = np.array([[np.transpose(feature) for feature in file_features] for file_features in files_features])
            return np.array([self._reshape_features_for_one_file(file_features) for file_features in tqdm(files_features)])

class Feature_Extractor_Training_Testing(Feature_Extractor):
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
            files = [wav_file for wav_file in files if wav_file[self.emotion_letter_position] in self.e_to_n_mapping.keys()]
            print(files[0])
            self.features = np.array([self._get_audio_features(wav_file) for wav_file in tqdm(files)])
            targets = [wav_file[self.emotion_letter_position] for wav_file in files]
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
            self.inputs, self.targets = self._shuffle_data(self.inputs, self.targets)
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

class Feature_Extractor_Inference(Feature_Extractor):
      def __init__(self, directory_name_list):
            super().__init__(directory_name_list)
      
      def get_features(self):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                Arguments:
                files - the list of file from which to extract the features
            """
            print("------------------ Extracting audio features from files")
            self.files = self.files[0]
            print(self.files[0])
            self.features = np.array([self._get_audio_features(wav_file) for wav_file in tqdm(self.files)])
            self.show_pic(self.features[0])
            self.features = self._reshape_features(self.features)
            print("------------------------------------------------------------------------")
            return self.features

#%%
class Data_Producer_Train_Test(object):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_Training_Testing(config.dir_name, config.data_set_name)
            self._train_test_slice = config.train_test_slice

      def _import_data(self):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
            """
            self._inputs, self._targets, self._feature_count = self._feature_extractor.get_featurs_and_targets()

      def _separate_train_from_test(self):
            """ REGROUP DATA INTO TRAIN DATA AND TEST DATA
                - given the small number of sample the validation phase is ignored
            """
            self._train_length = int(self._inputs.shape[0]*self._train_test_slice)
            self._test_length = int(self._inputs.shape[0]*(1-self._train_test_slice))

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
                lambda: self._train_inputs, tf.float64, output_shapes=[None, self._feature_count])
            self._train_targets_dt = tf.data.Dataset.from_generator(
                lambda: self._train_targets, tf.float64, output_shapes=[None])
            self._test_inputs_dt = tf.data.Dataset.from_generator(
                lambda: self._test_inputs, tf.float64, output_shapes=[None, self._feature_count])
            self._test_targets_dt = tf.data.Dataset.from_generator(
                lambda: self._test_targets, tf.float64, output_shapes=[None])

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
            X_test = iterator_train_inputs.get_next()
            y_test = iterator_train_targets.get_next()

            return (X_train, y_train), (X_test, y_test), (self._train_length, self._test_length)

#%%
class Data_Producer_Inference(object):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_Inference(config.dir_name)

      def _import_data(self):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
            """
            self._features = self._feature_extractor.get_features()

      def produce_data(self, session, name=None):
            """ CONSTRUCTING TF.DATASETS BASED ON THE FEATURES EXTRACTED
                Returns:
                (X_train, y_train) - pair representing one instance of the train data
                (X_test, y_test) - pair representing one instance of the train data
                (self._train_length, self._test_length) - pair representing the length of the train and test data                
            """
            self._import_data()
            
            inference_length = self._features.shape[0]
            feature_count = self._features[0].shape[1]
            self._features_dt = tf.data.Dataset.from_generator(lambda: self._features, tf.float64, output_shapes=[None, feature_count]).repeat()

            features = self._features_dt.make_one_shot_iterator()

            inputs = features.get_next()

            return inputs, inference_length
#%%
