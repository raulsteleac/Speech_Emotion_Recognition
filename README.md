# End-to-end Speech Emotion Recognition using BLSTMs with Attention layer and Multi-domain training 

This repository contains the source code for a Speech Emotion Recognition (SER) model built using Tensorflow 1.15 and a set of Python libraries that were used to create the different features of the project like: GUI, sound processing, recording and display of statistics.  

This project represents my undergraduate thesis and is only a mere starting point, basis, to which many improvements can be applied. 

  

  

![projPic](https://github.com/raulsteleac/Speech_Emotion_Recognition/blob/master/Sistem_Diagram.jpeg?raw=true) 

  

The SER architecture uses machine learning models for both feature extraction and classification in an end-to-end setting: 

  - For feature extraction I used two CNN layers combined with batch normalization. These were used to automatically detect relevant features from a visual representation of the sound, the Mel-spectrogram. 

  - The classification model consists of two BLSTM cells, that were used to take advantage of the temporal relationship between emotions at different time stamps, as the emotional information flows throughout the speech utterance. The attention layer was concatenated to the RNN in order to force the model to focus on the highly emotional frames and, therefore, disregard the ones that lack emotions and would work as noise. 

  

The user can train the model with different configurations and observe statistics (accuracy graph and confusion matrix) during training in the graphical user interface. For inference, the user can classify both pre-recorded speech utterances and ones that are recorded on the spot.  

## Project structure 

  

The classification model can be found in **model.py**, while the feature extraction is separated in the **feature_extractors/** directory. The feature extraction was implemented in two ways, hand-crafted and end-to-end, but only end-to-end is available for inference, as it gave better results with the current model. 

  

The graphical user interface was created using the PyQt5 library and the source code can be found in **graphics/** folder. The recording functionality was created using PyAudio and its implementation can be found in **recording/**.  

  

Other libraries used were: Librosa (audio processing), qdarkgraystyle (GUI theme), webrtcvad (speech detection during recording). 

  

In the directory **best_current_model**, one can find a pre-trained model that gave the best results in both testing and inference. However, this model is very sensitive to microphone and room settings and it still suffers from the obstacles of the SER field. 

  

To run the project, one should have both Tensorflow 1.15 and the necessary libraries installed in a Python enviornement. By simply using the command **"python SER.py"** one can start the graphical user interface and start training or inferring using the current model architecture. 
