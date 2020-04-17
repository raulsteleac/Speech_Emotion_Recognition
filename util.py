import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
from PyQt5 import QtCore
import os

import matplotlib.pyplot as plt

def get_files_from_directory(dir_name):
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

def one_hotizize(targets, target_class_vector, target_class_number):
            """ CONVERT THE LETTERS REPRESENTING EMOTIONS INTO ONE HOT ENCODING
                Arguments:
                targes - list of emotion coressponding to each input file
                Returns:
                The one-hot encoded version of the targets
            """
            targets = [target_class_vector[emotion] for emotion in targets]
            return np.eye(target_class_number)[targets]


def batch_normalization(batch):
      """ COMPUTE BATCH NORMALIZAION ON THE BATCH GIVEN AS INPUT TO EACH HIDDEN LAYER
            -Arguments:
                  batch: the batch of data to be normalized
            -Return:
                  Returns the transformed batch with mean 0 and stdev 1  
      """
      means = tf.reduce_mean(batch)
      stdev = tf.reduce_mean(np.power((batch - means), 2)) + 1e-6
      return (batch - means) / (tf.sqrt(stdev))
      # return tf.nn.batch_normalization(batch, mean = means, variance=stdev, offset=0, scale=1, variance_epsilon=1e-6)

def shuffle_data(inputs, targets):
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

def empty_dir(dir_name):
      onlyfiles = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
      for file_ in onlyfiles:
            os.remove(file_)

def show_pic(features, names, fig_size):
            """ LOG FUNCTION THAT PRINTS A PLOT FOR EACH FEATURE (MFCC, ZCR,.etc) OF ONE FILE
            """
            i = 1
            plt.figure(figsize=fig_size)
            for signal in features:
                  plt.subplot(np.shape(features)[0], 2, i)
                  plt.title(names[i-1])
                  librosa.display.specshow(signal)
                  plt.colorbar()
                  i = i+1
            plt.show()

seed_ = 1
def generator_shuffle(data, change):
      global seed_
      seed_ += change
      np.random.seed(seed_)
      splits_count = 1
      indexes = np.arange(splits_count)
      np.random.shuffle(indexes)
      for i in indexes:
            for j in range(data.shape[0] // splits_count):
                  yield data[i * (data.shape[0] // splits_count) + j]

class EMO_DB_Config(object):
      dir_name = ['data_sets/EMO-DB', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['EMO-DB', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']


class SAVEE_Config(object):
      dir_name = ['data_sets/SAVEE', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['SAVEE', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']


class RAVDESS_Config(object):
      dir_name = ['data_sets/RAVDESS', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['RAVDESS', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']

class ENTERFACE_Config(object):
      dir_name = ['data_sets/ENTERFACE', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['ENTERFACE', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']

class EMOVO_Config(object):
      dir_name = ['data_sets/EMOVO', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['EMOVO', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']

class MAV_Config(object):
      dir_name = ['data_sets/MONTREAL_AFFECTIVE_VOICE', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['MAV', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']

class MELD_Config(object):
      dir_name = ['data_sets/MELD', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['MELD', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']

class JL_Config(object):
      dir_name = ['data_sets/JL', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['JL', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']

class MULTIPLE_DATA_SETS_Config(object):
      dir_name = ['data_sets/EMO-DB', 'data_sets/RAVDESS', 'data_sets/EMOVO', 'data_sets/MONTREAL_AFFECTIVE_VOICE', 'data_sets/ENTERFACE', 'data_sets/JL', 'data_sets/Inregistrari_Proprii']
      data_set_name = ['EMO-DB', 'RAVDESS', 'EMOVO', 'MAV', 'ENTERFACE', 'JL', 'InrP']
      train_test_slice = 0.8
      target_domain = ['InrP']


class Inference_Config(object):
      dir_name = ['Inference']

def select_config(id_config):
      switcher = {
          1: EMO_DB_Config(),
          2: SAVEE_Config(),
          3: RAVDESS_Config(),
          4: ENTERFACE_Config(),
          5: EMOVO_Config(),
          6: MAV_Config(),
          7: MELD_Config(),
          8: JL_Config(),
          9: MULTIPLE_DATA_SETS_Config()
      }
      return switcher[id_config]
