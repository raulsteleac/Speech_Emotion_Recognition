#%%
import numpy as np
import atexit
import threading
import pyaudio
import wave
import webrtcvad
import sys

class MicrophoneRecorder(object):
      def __init__(self, rate=48000, chunksize=960):
            self.rate = rate
            self.chunksize = chunksize
            self.p = pyaudio.PyAudio()
            self.channels = 1
            self.sample_format = pyaudio.paInt16

            self.stream = self.p.open(format=self.sample_format,
                                    channels=self.channels,
                                    rate=self.rate,
                                    input=True,
                                    frames_per_buffer=self.chunksize,
                                    stream_callback=self.new_frame)
            self.lock = threading.Lock()
            self.stop = False
            self.frames = []
            self._print_frames = np.array([])
            self._print_frames_count = 0
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(2)
            atexit.register(self.close)
      
      def check_device_availability(self):
            try:
                  self.p.get_default_input_device_info()
            except IOError:
                self.thread.app_rnning.open_alert_dialog(
                    title="Missing input device alert", text="We could not identify any audio input device.", info="Please try and reconnec the device and restart the app.")
                return False
            return True

      def new_frame(self, data, frame_count, time_info, status):
            data = np.frombuffer(data, dtype=np.int16)
            with self.lock:
                  if self.vad.is_speech(data, self.rate):
                        self.frames.append(data)
                  print(np.array(self.frames).shape)
                  if self._print_frames_count == 10:
                        self.thread.print_recording_signal.emit(self._print_frames)
                        self._print_frames = np.array([])
                        self._print_frames_count = 0
                  else:
                        self._print_frames = np.concatenate((self._print_frames,data), axis=0)
                        self._print_frames_count+=1
                  if self.stop:
                        return None, pyaudio.paComplete
            return None, pyaudio.paContinue

      def get_frames(self):
            with self.lock:
                  frames = self.frames
                  return frames

      def save_to_wav(self, filename="output.wav"):
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.sample_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.get_frames()))
            wf.close()

      def start(self, thread):
            print("Starting recording")
            self.thread = thread
            self.stream.start_stream()

      def close(self):
            print("Finishing recording")
            with self.lock:
                  self.stop = True
            self.stream.close()
            self.p.terminate()

