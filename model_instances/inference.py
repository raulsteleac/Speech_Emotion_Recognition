from model import SER_Data_Producer, Speech_Emotion_Recognizer
from feature_extractors.end_to_end_data_producers import Data_Producer_End_to_End_Inference
from feature_extractors.hand_crafted_data_producers import Data_Producer_Hand_Crafted_Inference
from util import *

class SER_Inference_Model(object):
    session = None
    model = None
    files = []
    
    def init_model(self, dir_name="Inference"):
        self.session = tf.Session()
        inf_config = Inference_Config()
        inf_config.dir_name=[dir_name]
        ser_dp_inference = Data_Producer_End_to_End_Inference(inf_config)

        self.model = Speech_Emotion_Recognizer(model_op_name="Inference", is_training=False, is_inference=True)

        infr_inputs, inference_length, self.files = ser_dp_inference.produce_data(self.session)

        if inference_length == 0:
                return None, None, []

        self.model.set_inputs_targets_length(inputs=infr_inputs, op_length=inference_length)

        self.model.model()
        self.model.create_saver()
        self.model.restore_model(self.session, "./model/model.ckpt")
        self.model.model()

    def close_model(self):
        if self.session == None:
                return
        self.session.close()

    def inference(self, file_to_show):
        writer = tf.summary.FileWriter('./graphs', self.session.graph)
        merged_summaries = tf.summary.merge_all()
        list_vars = self.model.run_model(self.session, writer, merged_summaries, file_to_show=file_to_show)
        return list_vars