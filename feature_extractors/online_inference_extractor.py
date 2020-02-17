# %%
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'Speech_Emotion_Recognition'))
    print(os.getcwd())
except:
    pass

import librosa
import numpy as np
import tensorflow as tf
from util import *

from tqdm import tqdm
from feature_extractors.end_to_end_data_producers import Data_Producer_End_to_End
# %%
class Feature_Extractor(object):
      def _get_audio_features(self, wav_file):
            pass


class Online_Feature_Extractor_End_to_End(Feature_Extractor):
      def __init__(self):
            super().__init__()

      def reshape_frames(self, stft, window_length):
            """  RESHAPE THE SPECTOGRAM INTO WINDOWS OF SHAPE 128x128 
                  -Arguments:
                        stft : The spectogram of the audio signal
                  -Returns:
                        The reshaped signal that will be passed to the convolutional layer
            """
            stft = np.transpose(stft)
            window_nr = (stft.shape[0] // window_length + 1) * window_length
            pad_size = window_nr - stft.shape[0]
            stft = np.pad(stft, ((0, pad_size), (0, 1)), 'edge')
            conv_frames = np.array(([stft[i * window_length:(i+1) * window_length]
                                     for i in range(int(stft.shape[0]/(stft.shape[1]) + 1))]))
            return conv_frames[:, :, 0:window_length]

      def _get_audio_features(self, frames, org_rt):
            # signal = librosa.resample(np.array(frames), 16000, org_rt)
            librosa.core.time_to_frames
            stft = librosa.feature.melspectrogram(
                frames, n_fft=256, win_length=128, hop_length=32, center=False)
            return stft
      
      def get_features_from_frames(self, session ,frames, org_rt):
            self.features = np.array(self.reshape_frames(self._get_audio_features(frames, org_rt), 128))
            self.features = np.array([self.features, self.features])
            self.features = np.array([np.reshape(stft, (stft.shape[0], stft[0].shape[0],  stft[0].shape[1])) for stft in self.features])
            return self.features

class Online_Data_Producer_End_to_End_Inference(Data_Producer_End_to_End):
      def __init__(self):
            self._feature_extractor = Online_Feature_Extractor_End_to_End()

      def _import_data(self, session, frames, org_rt):
            self._features = self._feature_extractor.get_features_from_frames(session, frames, org_rt)

      def produce_data(self, session, frames, org_rt, name=None):
            """ CONSTRUCTING TF.DATASETS BASED ON THE FEATURES EXTRACTED
                    -Arguments:
                        session: the tf.Session() the model is running on
                    -Returns:
                        inputs - the features extracted from the convolutional layers
                        inference_length - the number of files in the inference folder
                        self._files - the names of the files in the inference folder to pretty print              
            """
            self._import_data(session, frames, org_rt)
            self._features = np.array([_inputs.reshape(
                [_inputs.shape[0], _inputs.shape[1], _inputs.shape[2], 1]) for _inputs in self._features])

            inference_length = self._features.shape[0]
            self._features_dt = tf.data.Dataset.from_generator(lambda: self._features, tf.float32, output_shapes=[
                                                               None, self._features[0].shape[1], self._features[0].shape[2], 1]).repeat()
            features = self._features_dt.make_one_shot_iterator()
            inputs = features.get_next()
            inputs = self._convolutional_feature_extractor(inputs)

            return inputs, inference_length
