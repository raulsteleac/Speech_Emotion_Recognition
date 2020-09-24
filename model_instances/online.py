from model import SER_Data_Producer, Speech_Emotion_Recognizer
from feature_extractors.online_inference_extractor import Online_Data_Producer_End_to_End_Inference
from feature_extractors.online_inference_extractor import Online_Data_Producer_Hand_Crafted_Inference
from util import *

class SER_Online_Model(object):
    session = None
    infr_inputs = None
    inference_length = None
    ses_online = None
    model = None
    def init_online_model(self):
        self.infr_inputs = tf.placeholder(tf.float32, (None, 256))
        self.inference_length = tf.placeholder(tf.float32, None)

        self.session = tf.Session()     
        self.model = Speech_Emotion_Recognizer(model_op_name="Online", is_training=False, is_inference=True)
        self.model.set_inputs_targets_length(inputs=self.infr_inputs, op_length=self.inference_length)

        self.model.model()
        self.model.create_saver()
        self.model.restore_model(self.session, "./model/model.ckpt")

        self.model.model()

    def online(self, frames, org_rt):
        writer = tf.summary.FileWriter('./graphs', self.session.graph)
        merged_summaries = tf.summary.merge_all()
        ser_dp_online = Online_Data_Producer_End_to_End_Inference()
        online_inputs_, online_length_ = ser_dp_online.produce_data(self.session, frames, org_rt)
        list_vars = self.model.run_model(self.session, writer, merged_summaries, feed_dict={
                                                    self.infr_inputs: self.session.run(online_inputs_), self.inference_length: online_length_})
        return list_vars
