# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SER_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'Speech_Emotion_Recognition'))
    print(os.getcwd())
except:
    pass

from PyQt5 import QtCore, QtGui, QtWidgets
from model import main, inference, online, init_inference_model, close_inference_model, init_online_model
from util import *
from pyqtgraph import PlotWidget
from recording.recorder import MicrophoneRecorder
import time

class Ui_MainWindow(object):
    accuracy_vals = []
    recording_vals = np.zeros([960 * 10 * 10, ])
    reg_count = 0
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setMinimumSize(QtCore.QSize(552, 0))
        self.groupBox_4.setObjectName("groupBox_4")

        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(30, 400, 400, 20))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_2.setObjectName("label_2")

        self.progressBar = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar.setGeometry(QtCore.QRect(30, 420, 500, 40))
        self.progressBar.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar.setStyleSheet("selection-background-color: rgb(184, 0, 0);")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setGeometry(QtCore.QRect(30, 460, 400, 20))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_3.setObjectName("label_3")

        self.progressBar_2 = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar_2.setGeometry(QtCore.QRect(30, 480, 500, 40))
        self.progressBar_2.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar_2.setStyleSheet("selection-background-color: rgb(248, 248, 0);")
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")

        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setGeometry(QtCore.QRect(30, 520, 400, 20))
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_4.setObjectName("label_4")

        self.progressBar_3 = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar_3.setGeometry(QtCore.QRect(30, 540, 500, 40))
        self.progressBar_3.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar_3.setStyleSheet("selection-background-color: rgb(168, 170, 165);")
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")

        self.label_5 = QtWidgets.QLabel(self.groupBox_4)
        self.label_5.setGeometry(QtCore.QRect(30, 580, 400, 20))
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_5.setObjectName("label_5")

        self.progressBar_4 = QtWidgets.QProgressBar(self.groupBox_4)
        self.progressBar_4.setGeometry(QtCore.QRect(30, 600, 500, 40))
        self.progressBar_4.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar_4.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")

        self.line_3 = QtWidgets.QFrame(self.groupBox_4)
        self.line_3.setGeometry(QtCore.QRect(10, 390, 531, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")

        self.graphicsView = PlotWidget(self.groupBox_4)
        self.graphicsView.setGeometry(QtCore.QRect(10, 30, 531, 361))
        self.graphicsView.setObjectName("graphicsView")                
        self.verticalLayout.addWidget(self.groupBox_4)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 3, 1, 1)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setMinimumSize(QtCore.QSize(0, 95))
        self.groupBox_3.setObjectName("groupBox_3")

        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_2.setGeometry(QtCore.QRect(100, 60, 441, 25))
        self.comboBox_2.setObjectName("comboBox_2")

        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_2.setGeometry(QtCore.QRect(100, 30, 441, 25))
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(10, 32, 80, 16))
        self.label_9.setObjectName("label_9")

        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(60, 64, 30, 15))
        self.label_10.setObjectName("label_10")

        self.pushButtonInfPlay = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButtonInfPlay.setGeometry(QtCore.QRect(550, 60, 45, 25))
        self.pushButtonInfPlay.setObjectName("pushButtonInfPlay")
        self.verticalLayout_2.addWidget(self.groupBox_3)

        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")        
        self.verticalLayout_2.addWidget(self.line, 0, QtCore.Qt.AlignTop)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 90))
        self.groupBox_2.setObjectName("groupBox_2")

        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 60, 160, 21))
        self.radioButton_3.setChecked(True)
        self.radioButton_3.setAutoRepeat(False)
        self.radioButton_3.setObjectName("radioButton_3")

        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QtCore.QRect(100, 60, 111, 21))
        self.radioButton_4.setObjectName("radioButton_4")

        self.pushButtonLoad = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButtonLoad.setGeometry(QtCore.QRect(250, 60, 100, 25))
        self.pushButtonLoad.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonLoad.setObjectName("pushButtonLoad")

        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(10, 30, 94, 23))
        self.label_8.setObjectName("label_8")
        
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(110, 30, 101, 25))
        self.lineEdit.setObjectName("lineEdit")

        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(320, 30, 221, 23))
        self.comboBox.setMinimumSize(QtCore.QSize(200, 0))
        self.comboBox.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.comboBox.setMaxVisibleItems(10)
        self.comboBox.setIconSize(QtCore.QSize(16, 16))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(220, 35, 101, 16))
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(30, 70))
        self.groupBox.setObjectName("groupBox")

        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(10, 30, 54, 21))
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")

        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(100, 30, 226, 21))
        self.radioButton_2.setObjectName("radioButton_2")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(250, 30, 100, 23))
        self.pushButton.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButton.setIconSize(QtCore.QSize(16, 16))
        self.pushButton.setObjectName("pushButton")

        self.pushButtonStop = QtWidgets.QPushButton(self.groupBox)
        self.pushButtonStop.setGeometry(QtCore.QRect(380, 30, 100, 23))
        self.pushButtonStop.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonStop.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonStop.setObjectName("pushButtonStop")
        self.verticalLayout_2.addWidget(self.groupBox)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setMinimumSize(QtCore.QSize(30, 100))
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_2.addWidget(self.groupBox_5)

        self.graphicsViewRec = PlotWidget(self.groupBox_5)
        self.graphicsViewRec.setGeometry(QtCore.QRect(115, 17, 480, 75))
        self.graphicsViewRec.setObjectName("graphicsViewRec")
        self.graphicsViewRec.setYRange(-20000, 20000,padding=0)
        self.graphicsViewRec.setXRange(0, 10 * 125 * 94, padding=0)

        self.pushButtonRecord = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonRecord.setGeometry(QtCore.QRect(10, 17, 100, 23))
        self.pushButtonRecord.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonRecord.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonRecord.setObjectName("pushButtonRecord")

        self.pushButtonStopRecord = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonStopRecord.setGeometry(QtCore.QRect(10, 43, 100, 23))
        self.pushButtonStopRecord.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonStopRecord.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonStopRecord.setObjectName("pushButtonStopRecord")

        self.pushButtonPlay = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonPlay.setGeometry(QtCore.QRect(10, 68, 100, 23))
        self.pushButtonPlay.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonPlay.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonPlay.setObjectName("pushButtonPlay")

        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMinimumSize(QtCore.QSize(600, 400))
        self.label_7.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.label_7.setStyleSheet("font: 9pt \"Sans Serif\";\n"
"background-color: rgb(0, 0, 0);")
        self.label_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_7.setLineWidth(4)
        self.label_7.setTextFormat(QtCore.Qt.AutoText)
        self.label_7.setScaledContents(True)
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_7.setWordWrap(True)
        self.label_7.setIndent(0)
        self.label_7.setOpenExternalLinks(False)
        self.label_7.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1212, 20))
        self.menubar.setObjectName("menubar")

        self.menuProject = QtWidgets.QMenu(self.menubar)
        self.menuProject.setObjectName("menuProject")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionReset = QtWidgets.QAction(MainWindow)
        self.actionReset.setObjectName("actionReset")
        
        self.menuProject.addAction(self.actionReset)
        self.menubar.addAction(self.menuProject.menuAction())

        self.retranslateUi(MainWindow)
        self.lineEdit_2.setText("Inference")
        self.lineEdit.setText("10")
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        fill_file(self)

        self.pushButton.clicked.connect(lambda: on_button_clicked(self))
        self.pushButtonStop.clicked.connect(lambda: on_buttonStop_clicked(self))
        self.pushButtonRecord.clicked.connect(lambda: on_buttonRecord_clicked(self))
        self.pushButtonPlay.clicked.connect(lambda: play_recording())
        self.pushButtonInfPlay.clicked.connect(lambda: play_recording(self.comboBox_2.currentText()))
        self.pushButtonStopRecord.clicked.connect(lambda: on_buttonStopRecord_clicked(self))
        self.lineEdit_2.textChanged.connect(lambda: fill_file(self))
        self.radioButton_2.toggled.connect(lambda: init_inference(self))
        self.print_accuracy_graph(0)

    def refresh_label_7(self):
        global nr
        nr = 0
        _translate = QtCore.QCoreApplication.translate
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#55ff7f;\"> ...</span></p></body></html>"))
    
    def refresh_graphics_view(self):
        self.graphicsView.clear()
        self.accuracy_vals = []
        self.print_accuracy_graph(0)

    def refresh_rec_graphics_view(self):
        self.recording_vals = np.zeros([960 * 10 * 10,])
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Statistics"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Recording"))
        self.label_2.setText(_translate("MainWindow", "Angry"))
        self.label_3.setText(_translate("MainWindow", "Happy"))
        self.label_4.setText(_translate("MainWindow", "Sad"))
        self.label_5.setText(_translate("MainWindow", "Neutral"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Pick a file to classify"))
        self.label_9.setText(_translate("MainWindow", "Folder path:"))
        self.label_10.setText(_translate("MainWindow", "File"))
        self.pushButtonInfPlay.setText(_translate("MainWindow", "Play"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Model settings"))
        self.radioButton_3.setText(_translate("MainWindow", "End-to-end"))
        self.radioButton_4.setText(_translate("MainWindow", "Hand-Crafted"))
        self.pushButtonLoad.setText(_translate("MainWindow", "Load"))
        self.label_8.setText(_translate("MainWindow", "Epochs:"))
        self.comboBox.setItemText(0, _translate("MainWindow", "EMO-DB"))
        self.comboBox.setItemText(1, _translate("MainWindow", "SAVE"))
        self.comboBox.setItemText(2, _translate("MainWindow", "RAVDESS"))
        self.comboBox.setItemText(3, _translate("MainWindow", "ENTERFACE"))
        self.comboBox.setItemText(4, _translate("MainWindow", "EMOVO"))
        self.comboBox.setItemText(5, _translate("MainWindow", "MAV"))
        self.comboBox.setItemText(6, _translate("MainWindow", "URDU"))
        self.comboBox.setItemText(7, _translate("MainWindow", "MULTIPLE"))
        self.label.setText(_translate("MainWindow", "Select dataset:"))
        self.groupBox.setTitle(_translate("MainWindow", "Actions"))
        self.radioButton.setToolTip(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.radioButton.setText(_translate("MainWindow", "Train"))
        self.radioButton_2.setText(_translate("MainWindow", "Inference"))
        self.pushButton.setText(_translate("MainWindow", "Start"))
        self.pushButtonStop.setText(_translate("MainWindow", "Stop"))
        self.pushButtonRecord.setText(_translate("MainWindow", "Record"))
        self.pushButtonPlay.setText(_translate("MainWindow", "Play"))
        self.pushButtonStopRecord.setText(_translate("MainWindow", "Stop"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#55ff7f;\"> ...</span></p></body></html>"))
        self.menuProject.setTitle(_translate("MainWindow", "Project"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.pushButtonRecord.setEnabled(False)
        self.pushButtonStopRecord.setEnabled(False)
        self.pushButtonStop.setEnabled(False)
        self.pushButtonPlay.setEnabled(False)
        self.pushButtonInfPlay.setEnabled(False)
        self.graphicsViewRec.setYRange(-20000, 20000, padding=0)
        self.graphicsViewRec.setXRange(0, 10 * 960 * 10, padding=0)
        self.graphicsViewRec.getPlotItem().hideButtons()
        self.graphicsViewRec.getPlotItem().hideAxis('left')
        self.graphicsViewRec.getPlotItem().hideAxis('bottom')

    def print_accuracy_graph(self, accuracy):
        self.accuracy_vals.append(accuracy)
        self.graphicsView.plot(self.accuracy_vals)

    def print_recording_graph(self, frames=None):
        self.graphicsViewRec.clear()
        self.recording_vals[0:9 * 960 * 10] = self.recording_vals[1 * 960 * 10:10 * 10 * 960]
        self.recording_vals[9 * 960 * 10:10 * 960 * 10] = frames
        self.graphicsViewRec.plot(self.recording_vals)
    
    def print_stats_model(self, string):
        print_in_label_7(self, string)

map_config = {
    "EMO-DB": 1,
    "SAVE": 2,
    "RAVDESS": 3,
    "ENTERFACE": 4,
    "EMOVO": 5,
    "MAV": 6,
    "URDU": 7,
    "MULTIPLE": 8,
}

def fill_file(app):
    files = get_files_from_directory(app.lineEdit_2.text())
    app.comboBox_2.clear()
    for file in files:
        app.comboBox_2.addItem(file)

ses = 0
ser_inference_model = 0
files = []

ses_online = 0
ses_online_model = 0
def init_inference(app):
    global ses, ser_inference_model, files, ses_online, ses_online_model
    if app.radioButton_2.isChecked():  # inference
        app.pushButtonRecord.setEnabled(True)
        app.pushButtonInfPlay.setEnabled(True)
        ses, ser_inference_model, files = init_inference_model(app.radioButton_3.isChecked())
        ses_online, ses_online_model = init_online_model()
    elif ses != 0 and app.radioButton.isChecked():
        app.pushButtonRecord.setEnabled(False)
        app.pushButtonInfPlay.setEnabled(False)
        app.pushButtonStopRecord.setEnabled(False)
        close_inference_model(ses)

thread_1 = 1
def on_button_clicked(app):
    global thread_1
    if app.radioButton.isChecked():  # training
        app.pushButton.setEnabled(False)
        app.pushButtonStop.setEnabled(True)
        app.refresh_label_7()
        app.refresh_graphics_view()
        thread_1 = Train_App(app)
        thread_1.print_accuracy_signal.connect(app.print_accuracy_graph)
        thread_1.print_stats.connect(app.print_stats_model)
        thread_1.start()
    elif app.radioButton_2.isChecked():  # inference
        global ses, ser_inference_model, files
        vals = inference(ses, ser_inference_model, files ,app.comboBox_2.currentText()) * 100
        app.progressBar.setValue(vals[0])
        app.progressBar_2.setValue(vals[1])
        app.progressBar_3.setValue(vals[2])
        app.progressBar_4.setValue(vals[3])
        print_in_label_7(app, str(list(map('{:.8f}'.format, vals))))
    pass

def on_buttonStop_clicked(app):
    if thread_1 != 1:
        thread_1.stopFlag = True
    pass

mr = None
recorder_thread = 1
def on_buttonRecord_clicked(app):
      app.refresh_rec_graphics_view()
      global mr, recorder_thread
      mr = MicrophoneRecorder()
      recorder_thread = Record_App(app, mr)
      recorder_thread.print_recording_signal.connect(app.print_recording_graph)
      recorder_thread.start()
      app.pushButtonStopRecord.setEnabled(True)

def on_buttonStopRecord_clicked(app):
        import librosa
        global ses,  mr, ses_online, ses_online_model
        mr.close()
        vals = []
        if np.array(mr.get_frames()).shape[0] > 30:
            app.pushButtonPlay.setEnabled(True)
            mr.save_to_wav()        
            frames, _ = librosa.load("output.wav", 16000)    
            vals = online(ses_online, ses_online_model, frames, 48000) * 100
        else:
            app.pushButtonPlay.setEnabled(False)
            vals = [0 for _ in range(4)]
        app.progressBar.setValue(vals[0])
        app.progressBar_2.setValue(vals[1])
        app.progressBar_3.setValue(vals[2])
        app.progressBar_4.setValue(vals[3])
        print_in_label_7(app, str(list(map('{:.8f}'.format, vals))))
        app.pushButtonRecord.setEnabled(True)
        app.pushButtonStopRecord.setEnabled(False)

nr = 0
def print_in_label_7(app, str):
    _translate = QtCore.QCoreApplication.translate
    global nr
    nr += 1
    if nr >= 26:
        txt = app.label_7.text().split("<html>")
        txt = "<html>".join(txt[1:26])
        app.label_7.setText(_translate(
            "MainWindow", txt + "<html><head/><body><span style=\" font-weight:600; color:#55ff7f;\">" + str + "</span></body></html>"))
    else:
        app.label_7.setText(_translate("MainWindow", app.label_7.text(
        ) + "<html><head/><body><span style=\" font-weight:600; color:#55ff7f;\">" + str + "</span></body></html>"))
import time


def play_recording(file="output.wav"):
    import wave
    import pyaudio
    chunk = 128

    f = wave.open(file, "rb")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=48000,
                    output=True)
    data = f.readframes(chunk)
    while data:
        stream.write(data)
        data = f.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

class Train_App(QtCore.QThread):
    print_accuracy_signal = QtCore.pyqtSignal(float)
    print_stats = QtCore.pyqtSignal(str)
    stopFlag = False
    def __init__(self, app_rnning, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.app_rnning = app_rnning
    
    def run(self):
        print("thread")
        main(self, int(self.app_rnning.lineEdit.text()),map_config[self.app_rnning.comboBox.currentText()], self.app_rnning.radioButton_3.isChecked())
        self.app_rnning.pushButton.setEnabled(True)
        self.app_rnning.pushButtonStop.setEnabled(False)


class Record_App(QtCore.QThread):
    print_recording_signal = QtCore.pyqtSignal(object)

    def __init__(self, app_rnning, mr, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.app_rnning = app_rnning
        self.mr = mr
        self.app_rnning.pushButtonRecord.setEnabled(False)

    def run(self):
        self.mr.start(self)
