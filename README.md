# Speech_Emotion_Recognition
The SER (Speech Emotion Recognition) model source is in the **model.py** who imports the different feature extraction methods (hand-crafted or end-to-end)
from the **/feature_extractors** directory.

Currently the model runs on pre-set parameters but a version that uses commnad line arguments and flags will be soon available.

The current inference method uses pre-recorded audio files placed in a specific folder, but a future goal is to realize online emotion recognition on ongoing conversations.
