import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os


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
      indexes = np.arange(data.shape[0])
      np.random.shuffle(indexes)
      for i in indexes:
            yield data[i]
