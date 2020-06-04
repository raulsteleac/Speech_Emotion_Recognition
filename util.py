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
      for root, _, files in os.walk(dir_name):
            for f in files:
                  if f.endswith(".wav"):
                              list_of_files = np.append(
                                  list_of_files, os.path.join(root, f))
      # print(list_of_files[0])
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

indexes_used = []
def init_indexes(train_length):
      global indexes_used
      indexes_used = np.arange(train_length)

def get_indexes():
      global indexes_used
      return indexes_used
def shuffle_indexes():
      global indexes_used
      np.random.shuffle(indexes_used)

def update_indexes(new_indexes):
      global indexes_used
      indexes_used = new_indexes

def generator_shuffle(data, idi, epochs):
      for i in get_indexes():
            yield data[i]

class EMO_DB_Config(object):
      dir_name = ['data_sets/EMO-DB']
      data_set_name = ['EMO-DB']


class SAVEE_Config(object):
      dir_name = ['data_sets/SAVEE']
      data_set_name = ['SAVEE']

class RAVDESS_Config(object):
      dir_name = ['data_sets/RAVDESS']
      data_set_name = ['RAVDESS']

class ENTERFACE_Config(object):
      dir_name = ['data_sets/ENTERFACE']
      data_set_name = ['ENTERFACE']

class EMOVO_Config(object):
      dir_name = ['data_sets/EMOVO']
      data_set_name = ['EMOVO']

class MAV_Config(object):
      dir_name = ['data_sets/MONTREAL_AFFECTIVE_VOICE']
      data_set_name = ['MAV']

class MELD_Config(object):
      dir_name = ['data_sets/MELD']
      data_set_name = ['MELD']

class JL_Config(object):
      dir_name = ['data_sets/JL']
      data_set_name = ['JL']


class Inrp_Config(object):
      dir_name = ['data_sets/Inregistrari_Proprii']
      data_set_name = ['InrP']

class MULTIPLE_DATA_SETS_Config(object):      
      dir_name = ['data_sets/EMO-DB', 'data_sets/RAVDESS', #'data_sets/SAVEE',  
      'data_sets/EMOVO',
       'data_sets/MONTREAL_AFFECTIVE_VOICE', 'data_sets/ENTERFACE', 
       'data_sets/JL', 
       'data_sets/Inregistrari_Proprii']
      data_set_name = ['EMO-DB', 'RAVDESS', #'SAVEE', 
      'EMOVO', 'MAV', 'ENTERFACE', 'JL', 'InrP']


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
          9: Inrp_Config(),
          10: MULTIPLE_DATA_SETS_Config()
      }
      return switcher[id_config]
