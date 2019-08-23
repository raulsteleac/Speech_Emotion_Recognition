#%%
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'Speech_Emotion_Recognition'))
    print(os.getcwd())
except:
    pass

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

class Feature_Extractor_Auto(object):
      def __init__(self, directory_name_list, data_set_name_list):
            print("------------------ Starting extracting featuers for data set : %s " % ' '.join(data_set_name_list))
            self.files = [self._get_files_from_directory(directory_name) for directory_name in directory_name_list]
            self._data_set_name_list = data_set_name_list

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

      def reshape_framse(self, stft):
            window_length = 128
            stft = np.transpose(stft)
            window_nr = (stft.shape[0] // window_length + 1) * window_length
            pad_size = window_nr - stft.shape[0]
            stft = np.pad(stft, ((0, pad_size), (0, 0)), 'edge')
            conv_frames = np.array([stft[i* 128 :(i+1) * 128] for i in range(int(stft.shape[0]/(stft.shape[1])+1))])
            return conv_frames[:,:,0:128]
            
      def _get_audio_features(self, wav_file):
            """ EXTRACT THE AUDIO FEATURES FROM THE .WAV FILES USING THE LIBROSA LIBRARY
                Arguments:
                wav_file - the name of the .wav file from which to extract the features
                Local variables:
                stft - the coefficients of the Mel Frequency Cepstral for each frame
            """
            signal, rate = librosa.load(wav_file, 16000)
            librosa.core.time_to_frames
            stft = np.abs(librosa.stft(signal,n_fft=256, win_length=128, hop_length=32, center=False))
            return stft

      def _one_hotizize(self, targets):
            """ CONVERT THE LETTERS REPRESENTING EMOTIONS INTO ONE HOT ENCODING
                Arguments:
                targes - list of emotion coressponding to each input file
                Returns:
                The one-hot encoded version of the targets
            """
            targets = [self.e_to_n_mapping[emotion] for emotion in targets]
            return np.eye(self.emotion_number)[targets]

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

      def _transform_wave_files(self, files):
            """ CALL THE FEATURE EXTRACTION FUNCTIONS ON ALL FILES IN THE DATA SET  
                Arguments:
                files - the list of file from which to extract the features
            """
            print("------------------ Extracting audio features from files")
            files = [wav_file for wav_file in files if wav_file[self.emotion_letter_position] in self.e_to_n_mapping.keys()]
            print(files[0])
            self.features = np.array([self.reshape_framse(self._get_audio_features(wav_file)) for wav_file in tqdm(files)])
            self.feature_print = self._get_audio_features(files[0])
            targets = [wav_file[self.emotion_letter_position] for wav_file in files]
            self.targets = np.append(self.targets, self._one_hotizize(targets))
      
      def get_featurs_and_targets(self):
            """ THIS FUNCTION WILL BE THE ONE CALLED FROM THE OUTSIDE OF THIS CLASS
                TO OBTAIN THE FEATURES AND TARGETS 
                Returns:
                self.inputs, self.targets - represents the list of features and targets propagated outside this class 
            """
            print("------------------ Processing audio files")
            self.targets = np.array([[]])

            for files, ds_name in zip(self.files, self._data_set_name_list):
                self._set_data_set_config(ds_name)
                self._transform_wave_files(files)
                self.show_pic(self.feature_print)
                self.inputs = np.array([np.reshape(stft, (stft.shape[0], stft[0].shape[0],  stft[0].shape[1], 1)) for stft in self.features])
            
            self.targets = np.reshape(self.targets, (-1, self.emotion_number))
            return self.inputs, self.targets, [None, self.inputs[0].shape[1], self.inputs[0].shape[2], 1]

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

class EMO_DB_Config(object):
      dir_name = ['EMO-DB']
      data_set_name = ['EMO-DB']
      train_test_slice = 0.8

class Data_Producer_Train_Test_Auto(object):
      def __init__(self, config):
            self._feature_extractor = Feature_Extractor_Auto(config.dir_name, config.data_set_name)
            self._train_test_slice = config.train_test_slice

      def _import_data(self):
            """ CALL OF THE GET FUNCTION OF THE FEATURE EXTRACTOR  
            """
            self._inputs, self._targets, self.feature_shape = self._feature_extractor.get_featurs_and_targets()

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

      def conv_layer(self, input_data, filter_size, channels_in, channels_out, strides, name="Conv"):
            W = tf.get_variable("Weights_Fully_Connected_Layer"+name,dtype=tf.float32, shape=[filter_size, filter_size, channels_in, channels_out])
            b = tf.get_variable("Biases_Fully_Connected_Layer"+name,dtype=tf.float32, shape=[channels_out])
            return tf.nn.conv2d(input=input_data, filter=W, strides=strides, padding='SAME') + b

      def _convolutional_feature_extractor(self, stft):
        self.init = tf.random_normal_initializer(-0.1, 0.1)
        with tf.variable_scope("Convbb", reuse=tf.AUTO_REUSE, initializer=self.init):
            #stft = tf.reshape(stft, (stft.shape[0], stft[0].shape[0],  stft[0].shape[1], 1))
            conv1 = self.conv_layer(input_data= stft, filter_size=128, channels_in=1, channels_out=32, strides=[1,2,2,1],name="conv1")
            conv2 = self.conv_layer(input_data= conv1, filter_size=64, channels_in=32, channels_out=32, strides=[1,2,2,1],name="conv2")
            conv3 = self.conv_layer(input_data= conv2, filter_size=64, channels_in=32, channels_out=32, strides=[1,1,1,1],name="conv3")
            conv4 = self.conv_layer(input_data= conv3, filter_size=32, channels_in=32, channels_out=32, strides=[1,8,4,1],name="conv4")
            conv_out = tf.reshape(conv4, (-1, 1024))
            outputs = tf.layers.dense(conv_out, 1024, activation=tf.nn.relu, name="Forth_Fully_Connected_Layer" )
        return outputs

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
            X_test = iterator_train_inputs.get_next()
            y_test = iterator_train_targets.get_next()

            X_train = self._convolutional_feature_extractor(X_train)
            X_test = self._convolutional_feature_extractor(X_test)
            session.run(tf.global_variables_initializer())
            return (X_train, y_train), (X_test, y_test), (self._train_length, self._test_length)
