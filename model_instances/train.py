from model import SER_Data_Producer, Speech_Emotion_Recognizer
from util import *

def main(thread=None, epochs=10, keep_prob=0.5, train_ratio = 0.8, lr = 0.0001, id_config=1, flag_end_to_end = 1):
      tf.reset_default_graph()
      ses = tf.Session()
      
      ser_dp = SER_Data_Producer(select_config(id_config), train_ratio, flag_end_to_end=flag_end_to_end, thread=thread)
      ser_dp.import_data(ses)

      train_inputs, train_targets, train_length = ser_dp.train_data
      test_inputs, test_targets, test_length = ser_dp.test_data

      ser_train_model = Speech_Emotion_Recognizer( "Training", keep_prob, lr, True, flag_end_to_end=flag_end_to_end)
      ser_test_model  = Speech_Emotion_Recognizer( "Testing",  flag_end_to_end=flag_end_to_end)

      ser_train_model.set_inputs_targets_length(train_inputs, train_targets, train_length)
      ser_test_model.set_inputs_targets_length(test_inputs, test_targets, test_length)

      init_indexes(train_length)
      ser_train_model.init_examples(get_indexes())

      ser_train_model.model()
      ser_test_model.model()

      ser_train_model.create_saver()

      writer = tf.summary.FileWriter('./graphs', ses.graph)
      merged_summaries = tf.summary.merge_all()

      print("\n %s just started ! \n" % ser_train_model.model_op_name)
      ser_train_model.initialize_variables(ses)
      for epoch in range(epochs):
            ser_train_model.refresh_current_examp()
            # shuffle_indexes()
            print("-----------> Epoch " + str(epoch))
            if epoch % 5 == 0 and epoch and thread.app_rnning.ooda_check_box.isChecked():
                  ser_train_model.update_input_length(epochs, epoch, float(thread.app_rnning.horizontalSlider_ooda.value() / 10))
                  ser_train_model.calculate_worst_input_examples()
                  update_indexes(ser_train_model.get_keys())
                  ser_train_model.init_examples(get_indexes())

            if thread:
                  thread.print_epoch.emit(str(epoch))
            x = ser_train_model.run_model(ses, writer, merged_summaries, thread=thread)
            if x == 1:
                  break
            if (epoch) % 5 == 0:
                  print("----------------------------------------------------------------")
                  ser_test_model.run_model(ses, writer, merged_summaries, thread=thread, validation=True)
                  print(
                      "----------------------------------------------------------------")
      empty_dir("./model")
      ser_train_model.save_model(ses, "./model/model.ckpt")
      writer = tf.summary.FileWriter('./graphs', ses.graph)
      thread.stopFlag = 0
      thread.print_stats.emit("############### Trainig finished!")
      print("\n %s just started ! \n" % ser_test_model.model_op_name)
      ser_test_model.run_model(ses, writer, merged_summaries, thread=thread)

      ses.close()