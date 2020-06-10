# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SER_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!
from pydub.playback import play
import os
try:
    os.chdir('/home/raulslab/work/Speech_Emotion_Recognition')
    print(os.getcwd())
except:
      print("Can't change the Current Working Directory")
      pass

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication
from model import main, inference, online, init_inference_model, close_inference_model, init_online_model
from util import *
from pyqtgraph import PlotWidget
from recording.recorder import MicrophoneRecorder
import time
import sys

COMPLETED_STYLE_ANGRY = """
QProgressBar {
    border: 1px solid #76797C;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: red;
}
"""

COMPLETED_STYLE_HAPPY = """
QProgressBar {
    border: 1px solid #76797C;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: yellow;
}
"""

COMPLETED_STYLE_SAD= """
QProgressBar {
    border: 1px solid #76797C;
    border-radius: 5px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: gray;
}
"""
SLYDER_ENABLED = """
QSlider::groove:horizontal {
    border: 1px solid #565a5e;
    height: 4px;
    background: #565a5e;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #D1DBCB;
    border: 1px solid #999999;
    width: 10px;
    height: 10px;
    margin: -5px 0;
}

QSlider::add-page:qlineargradient {
    background: #595858;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}

QSlider::sub-page::qlineargradient:horizontal {
    background:  #D1DBCB;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    border-top-left-radius: 5px;
    border-bottom-left-radius: 5px;
}"""
SLYDER_DISABLED = """

QSlider::groove:horizontal {
    border: 1px solid #565a5e;
    height: 4px;
    background: #595858;
    margin: 0px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #595858;
    border: 1px solid #999999;
    width: 10px;
    height: 10px;
    margin: -5px 0;
}

QSlider::add-page:qlineargradient {
    background: #595858;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
}

QSlider::sub-page::qlineargradient:horizontal {
    background:  #595858;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    border-top-left-radius: 5px;
    border-bottom-left-radius: 5px;
}"""

class Ui_MainWindow(object):
    accuracy_vals = []
    recording_vals = np.zeros([960 * 10 * 10, ])
    reg_count = 0
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.resize(1300, 900)

        qr = MainWindow.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        MainWindow.move(qr.topLeft())

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setMinimumSize(QtCore.QSize(670, 860))
        self.groupBox_4.setObjectName("groupBox_4")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 400, 400, 20))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_2.setObjectName("label_2")

        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(40, 400, 500, 40))
        self.progressBar.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar.setStyleSheet(COMPLETED_STYLE_ANGRY)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 460, 400, 20))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_3.setObjectName("label_3")

        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setGeometry(QtCore.QRect(40, 460, 500, 40))
        self.progressBar_2.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar_2.setStyleSheet(COMPLETED_STYLE_HAPPY)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 520, 400, 20))
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_4.setObjectName("label_4")

        self.progressBar_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3.setGeometry(QtCore.QRect(40, 520, 500, 40))
        self.progressBar_3.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar_3.setStyleSheet(COMPLETED_STYLE_SAD)
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 570, 400, 20))
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_5.setObjectName("label_5")

        self.progressBar_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_4.setGeometry(QtCore.QRect(40, 560, 500, 40))
        self.progressBar_4.setMinimumSize(QtCore.QSize(500, 40))
        self.progressBar_4.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")

        self.graphicsView = PlotWidget(self.groupBox_4)
        self.graphicsView.setGeometry(QtCore.QRect(10, 30, 650, 400))
        self.graphicsView.setObjectName("graphicsView")                
        self.verticalLayout.addWidget(self.groupBox_4)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 3, 1, 1)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setMinimumSize(QtCore.QSize(603, 110))
        self.groupBox_3.setObjectName("groupBox_3")

        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_2.setGeometry(QtCore.QRect(105, 72, 441, 30))
        self.comboBox_2.setObjectName("comboBox_2")

        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_2.setGeometry(QtCore.QRect(105, 35, 441, 30))
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(10, 37, 90, 30))
        self.label_9.setObjectName("label_9")

        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(60, 71, 40, 30))
        self.label_10.setObjectName("label_10")

        self.pushButtonInfPlay = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButtonInfPlay.setGeometry(QtCore.QRect(550, 72, 48, 30))
        self.pushButtonInfPlay.setObjectName("pushButtonInfPlay")
        
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMinimumSize(QtCore.QSize(600, 280))
        self.groupBox_2.setObjectName("groupBox_2")

        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setGeometry(QtCore.QRect(10, 75, 200, 25))
        self.label_18.setObjectName("label_18")
        
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(210, 77, 165, 25))
        self.radioButton_3.setChecked(True)
        self.radioButton_3.setAutoRepeat(False)
        self.radioButton_3.setObjectName("radioButton_3")

        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QtCore.QRect(320, 77, 120, 25))
        self.radioButton_4.setObjectName("radioButton_4")

        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(10, 243, 94, 25))
        self.label_8.setObjectName("label_8")
        
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(75, 240, 70, 30))
        self.lineEdit.setObjectName("lineEdit")

        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        self.label_17.setGeometry(QtCore.QRect(460, 243, 105, 30))
        self.label_17.setObjectName("label_17")

        self.label_19 = QtWidgets.QLabel(self.groupBox_2)
        self.label_19.setGeometry(QtCore.QRect(570, 243, 30, 30))
        self.label_19.setObjectName("label_19")

        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(125, 38, 460, 30))
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
        self.comboBox.addItem("")

        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(10, 36, 105, 30))
        self.label.setObjectName("label")

        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setGeometry(QtCore.QRect(10, 100, 125, 30))
        self.label_11.setObjectName("label_11")
        self.horizontalSlider = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider.setGeometry(QtCore.QRect(135, 109, 430, 17))
        self.horizontalSlider.setMaximum(10)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setGeometry(QtCore.QRect(570, 102, 31, 30))
        self.label_12.setObjectName("label_12")

        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setGeometry(QtCore.QRect(10, 132, 121, 25))
        self.label_15.setObjectName("label_15")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(135, 137, 430, 17))
        self.horizontalSlider_2.setMaximum(10)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setGeometry(QtCore.QRect(570, 134, 31, 25))
        self.label_16.setObjectName("label_15")


        self.ooda_check_box = QtWidgets.QCheckBox(self.groupBox_2)
        self.ooda_check_box.setGeometry(QtCore.QRect(15, 160, 120, 31))
        self.ooda_check_box.setObjectName("checkBoxOODA")
        self.horizontalSlider_ooda = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_ooda.setGeometry(QtCore.QRect(135, 168, 430, 17))
        self.horizontalSlider_ooda.setMaximum(9)
        self.horizontalSlider_ooda.setMinimum(1)
        self.horizontalSlider_ooda.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_ooda.setObjectName("horizontalSlider_ooda")
        self.label_ooda = QtWidgets.QLabel(self.groupBox_2)
        self.label_ooda.setGeometry(QtCore.QRect(570, 164, 31, 25))
        self.label_ooda.setObjectName("label_ooda")

        self.label_13 = QtWidgets.QLabel(self.groupBox_2)
        self.label_13.setGeometry(QtCore.QRect(10, 197, 121, 25))
        self.label_13.setObjectName("label_13")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox.setGeometry(QtCore.QRect(121, 197, 100, 31))
        self.doubleSpinBox.setDecimals(5)
        self.doubleSpinBox.setSingleStep(1e-05)
        self.doubleSpinBox.setObjectName("doubleSpinBox")

        self.verticalLayout_2.addWidget(self.groupBox_2)
        
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(30, 70))
        self.groupBox.setObjectName("groupBox")

        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(10, 30, 60, 30))
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")

        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(100, 30, 226, 30))
        self.radioButton_2.setObjectName("radioButton_2")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(250, 30, 100, 30))
        self.pushButton.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButton.setIconSize(QtCore.QSize(16, 16))
        self.pushButton.setObjectName("pushButton")

        self.pushButtonStop = QtWidgets.QPushButton(self.groupBox)
        self.pushButtonStop.setGeometry(QtCore.QRect(380, 30, 100, 30))
        self.pushButtonStop.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonStop.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonStop.setObjectName("pushButtonStop")
        self.verticalLayout_2.addWidget(self.groupBox)

        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setMinimumSize(QtCore.QSize(30, 155))
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.verticalLayout_2.addWidget(self.groupBox_5)

        self.graphicsViewRec = PlotWidget(self.groupBox_5)
        self.graphicsViewRec.setGeometry(QtCore.QRect(115, 37, 480, 110))
        self.graphicsViewRec.setObjectName("graphicsViewRec")
        self.graphicsViewRec.setYRange(-20000, 20000,padding=0)
        self.graphicsViewRec.setXRange(0, 10 * 125 * 94, padding=0)

        self.pushButtonRecord = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonRecord.setGeometry(QtCore.QRect(10, 37, 100, 30))
        self.pushButtonRecord.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonRecord.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonRecord.setObjectName("pushButtonRecord")

        self.pushButtonStopRecord = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonStopRecord.setGeometry(QtCore.QRect(10, 77, 100, 30))
        self.pushButtonStopRecord.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonStopRecord.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonStopRecord.setObjectName("pushButtonStopRecord")

        self.pushButtonPlay = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButtonPlay.setGeometry(QtCore.QRect(10, 117, 100, 30))
        self.pushButtonPlay.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonPlay.setIconSize(QtCore.QSize(16, 16))
        self.pushButtonPlay.setObjectName("pushButtonPlay")

        self.tabs = QtWidgets.QTabWidget(self.groupBox_4)
        self.tabs.setGeometry(QtCore.QRect(10, 440, 600, 400))
        self.tabs.setMinimumHeight(400)
        self.tabs.setMinimumWidth(600)
        self.tabs.setIconSize(QtCore.QSize(30, 300))
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(
            ["Angry", "Happy", "Sad", "Normal", " Total "])  # 
        self.tableWidget.setVerticalHeaderLabels(
            ["Angry", "Happy", "Sad", "Normal", "Total"])  #

        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(0, 3, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(0, 4, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(1, 0, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(1, 1, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(1, 2, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(1, 3, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(1, 4, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(2, 0, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(2, 1, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(2, 2, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(2, 3, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(2, 4, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(3, 0, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(3, 1, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(3, 2, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(3, 3, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(3, 4, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(4, 0, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(4, 1, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(4, 2, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(4, 3, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.setItem(4, 4, QtWidgets.QTableWidgetItem("0"))
        self.tableWidget.item(4, 4).setBackground(QtGui.QColor(102, 140, 255))
        self.tableWidget.item(3, 3).setBackground(QtGui.QColor(125, 125, 125))
        self.tableWidget.item(2, 2).setBackground(QtGui.QColor(125, 125, 125))
        self.tableWidget.item(1, 1).setBackground(QtGui.QColor(125, 125, 125))
        self.tableWidget.item(0, 0).setBackground(QtGui.QColor(125, 125, 125))

        for i in range(5):
            for j in range(5):
                self.tableWidget.item(i, j).setFlags(QtCore.Qt.ItemIsEnabled)
        self.tabs.resize(650, 425)

        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setMinimumSize(QtCore.QSize(600, 425))
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

        self.verticalLayoutTable = QtWidgets.QVBoxLayout()
        self.verticalLayoutTable.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayoutTable.setContentsMargins(33, 95, 33, 70)
        self.verticalLayoutTable.setSpacing(9)
        self.verticalLayoutTable.setObjectName("verticalLayoutTable")

        self.label_total = QtWidgets.QLabel(self.groupBox_4)
        self.label_total.setObjectName("label_nr")
        
        self.tab2 = QtWidgets.QWidget()
        self.tabs.addTab(self.label_7, "Logs")
        self.tabs.addTab(self.tab2, "Confusion matrix")

        self.tab2.layout = self.verticalLayoutTable
        self.tab2.layout.addWidget(self.tableWidget)
        self.tab2.layout.addWidget(self.label_total)
        self.tab2.setLayout(self.tab2.layout)

        self.verticalLayout_2.addWidget(self.label_2)
        self.verticalLayout_2.addWidget(self.progressBar)
        self.verticalLayout_2.addWidget(self.label_3)
        self.verticalLayout_2.addWidget(self.progressBar_2)
        self.verticalLayout_2.addWidget(self.label_4)
        self.verticalLayout_2.addWidget(self.progressBar_3)
        self.verticalLayout_2.addWidget(self.label_5)
        self.verticalLayout_2.addWidget(self.progressBar_4)
        
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.actionReset = QtWidgets.QAction(MainWindow)
        self.actionReset.setObjectName("actionReset")

        self.retranslateUi(MainWindow)
        self.lineEdit_2.setText("Inference")
        self.lineEdit.setText("10")
        self.horizontalSlider_2.setValue(5)
        change_label_16(self)
        self.horizontalSlider.setValue(8)
        change_label_12(self)
        self.horizontalSlider_ooda.setValue(8)
        change_label_ooda(self)
        self.doubleSpinBox.setValue(0.0001)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        fill_file(self)
        self.ooda_check_box.setChecked(False)
        self.horizontalSlider_ooda.setStyleSheet(SLYDER_DISABLED)
        self.horizontalSlider_ooda.setEnabled(False)

        self.pushButton.clicked.connect(lambda: on_start_button_clicked(self))
        self.pushButtonStop.clicked.connect(lambda: on_buttonStop_clicked(self))
        self.pushButtonRecord.clicked.connect(lambda: on_buttonRecord_clicked(self))
        self.pushButtonPlay.clicked.connect(lambda: play_recording(self))
        self.pushButtonInfPlay.clicked.connect(lambda: play_recording(self, self.comboBox_2.currentText()))
        self.pushButtonStopRecord.clicked.connect(lambda: on_buttonStopRecord_clicked(self))
        self.lineEdit_2.returnPressed.connect(lambda: fill_file(self))
        self.radioButton_2.toggled.connect(lambda: init_inference(self))
        self.horizontalSlider.valueChanged.connect(lambda: change_label_12(self))
        self.horizontalSlider_2.valueChanged.connect(lambda: change_label_16(self))
        self.horizontalSlider_ooda.valueChanged.connect(lambda: change_label_ooda(self))
        self.ooda_check_box.stateChanged.connect(lambda: change_horizontal_ooda(self))
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
        MainWindow.setWindowTitle(_translate("MainWindow", "Speech Emotion Recognizer"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Statistics"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Recording"))
        self.label_2.setText(_translate("MainWindow", "Angry"))
        self.label_3.setText(_translate("MainWindow", "Happy"))
        self.label_4.setText(_translate("MainWindow", "Sad"))
        self.label_5.setText(_translate("MainWindow", "Neutral"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Pick a file to classify"))
        self.label_9.setText(_translate("MainWindow", "Folder path:"))
        self.label_10.setText(_translate("MainWindow", "File:"))
        self.pushButtonInfPlay.setText(_translate("MainWindow", "Play"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Model settings"))
        self.radioButton_3.setText(_translate("MainWindow", "End-to-end"))
        self.radioButton_4.setText(_translate("MainWindow", "Hand-Crafted"))
        self.label_8.setText(_translate("MainWindow", "Epochs:"))
        self.label_18.setText(_translate("MainWindow", "Feature extractio tehnioque:"))
        self.label_11.setText(_translate("MainWindow", "Train / Test Ratio:"))
        self.label_12.setText(_translate("MainWindow", "1"))
        self.label_13.setText(_translate("MainWindow", "Learning Rate:"))
        self.label_15.setText(_translate("MainWindow", "Dropout Rate:"))
        self.label_16.setText(_translate("MainWindow", "1"))
        self.label_17.setText(_translate("MainWindow", "Current epoch:"))
        self.label_19.setText(_translate("MainWindow", "0"))
        self.label_ooda.setText(_translate("MainWindow", "0"))
        self.label_total.setText(_translate("MainWindow", "Numarul total de intrari = 0"))
        self.comboBox.setItemText(0, _translate("MainWindow", "EMO-DB"))
        self.comboBox.setItemText(1, _translate("MainWindow", "SAVEE"))
        self.comboBox.setItemText(2, _translate("MainWindow", "RAVDESS"))
        self.comboBox.setItemText(3, _translate("MainWindow", "ENTERFACE"))
        self.comboBox.setItemText(4, _translate("MainWindow", "EMOVO"))
        self.comboBox.setItemText(5, _translate("MainWindow", "MAV"))
        self.comboBox.setItemText(6, _translate("MainWindow", "MELD"))
        self.comboBox.setItemText(7, _translate("MainWindow", "JL"))
        self.comboBox.setItemText(7, _translate("MainWindow", "INRP"))
        self.comboBox.setItemText(8, _translate("MainWindow", "MULTIPLE"))
        self.ooda_check_box.setText(_translate("MainWindow", "OODA loop"))
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
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.pushButtonRecord.setEnabled(False)
        self.pushButtonStopRecord.setEnabled(False)
        self.pushButtonStop.setEnabled(False)
        self.pushButtonPlay.setEnabled(False)
        self.pushButtonInfPlay.setEnabled(False)

        self.progressBar.setEnabled(False)
        self.progressBar_2.setEnabled(False)
        self.progressBar_3.setEnabled(False)
        self.progressBar_4.setEnabled(False)
        self.label_2.setEnabled(False)
        self.label_3.setEnabled(False)
        self.label_4.setEnabled(False)
        self.label_5.setEnabled(False)
        self.groupBox_3.setEnabled(False)
        self.groupBox_5.setEnabled(False)

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

    def print_label_19(self, epoch):
            self.label_19.setText(epoch)


    def print_accuracy_matrix(self, matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(matrix[i][j])))
        for i in range(4):
            self.tableWidget.setItem(4, i, QtWidgets.QTableWidgetItem(str(np.sum(matrix[:,i]))))
            self.tableWidget.setItem(i, 4, QtWidgets.QTableWidgetItem(str(np.sum(matrix[i]))))
        self.tableWidget.setItem(4, 4, QtWidgets.QTableWidgetItem(str(np.sum(np.diag(matrix)))))
        self.tableWidget.item(4, 4).setBackground(QtGui.QColor(102, 140, 255))
        self.tableWidget.item(3, 3).setBackground(QtGui.QColor(125, 125, 125))
        self.tableWidget.item(2, 2).setBackground(QtGui.QColor(125, 125, 125))
        self.tableWidget.item(1, 1).setBackground(QtGui.QColor(125, 125, 125))
        self.tableWidget.item(0, 0).setBackground(QtGui.QColor(125, 125, 125))
        for i in range(5):
            for j in range(5):
                self.tableWidget.item(i, j).setFlags(QtCore.Qt.ItemIsEnabled)

    def open_alert_dialog(self, title="Alert", text="...", info="..."):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)

        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

map_config = {
    "EMO-DB": 1,
    "SAVEE": 2,
    "RAVDESS": 3,
    "ENTERFACE": 4,
    "EMOVO": 5,
    "MAV": 6, 
    "MELD": 7,
    "JL": 8,
    "INRP": 9,
    "MULTIPLE": 10,
}

ses = 0
ser_inference_model = 0
files = []

ses_online = 0
ses_online_model = 0

def change_label_16(app):
    app.label_16.setText(str(float(app.horizontalSlider_2.value())/10))

def change_label_12(app):
    app.label_12.setText(str(float(app.horizontalSlider.value())/10))

def change_label_ooda(app):
    app.label_ooda.setText(str(float(app.horizontalSlider_ooda.value())/10))


def change_horizontal_ooda(app):
    if app.ooda_check_box.isChecked():
        app.horizontalSlider_ooda.setStyleSheet(SLYDER_ENABLED)
        app.horizontalSlider_ooda.setEnabled(True)
    else:
        app.horizontalSlider_ooda.setStyleSheet(SLYDER_DISABLED)
        app.horizontalSlider_ooda.setEnabled(False)
    app.label_ooda.setEnabled(app.ooda_check_box.isChecked())
    

def fill_file(app):
    global ses, ser_inference_model, files, ses_online, ses_online_model
    files = get_files_from_directory(app.lineEdit_2.text())
    app.comboBox_2.clear()
    for file in files:
        app.comboBox_2.addItem(file)
    if app.radioButton_2.isChecked() :  # inference
        app.pushButton.setEnabled(True)
        app.pushButtonInfPlay.setEnabled(True)
        ses, ser_inference_model, files = init_inference_model(app.radioButton_3.isChecked(), app.lineEdit_2.text())
        if ser_inference_model == None:
            app.open_alert_dialog(title="Missing Inference Files Alert", text="We could no find any files to classify in the stated folder.", info="You can continue the inference process by using the online model.")
            app.pushButton.setEnabled(False)
            app.pushButtonInfPlay.setEnabled(False)
        ses_online, ses_online_model = init_online_model()
    
def init_inference(app):
    global ses, ser_inference_model, files, ses_online, ses_online_model
    if app.radioButton_2.isChecked():  # inference        
        if app.radioButton_4.isChecked():
            app.open_alert_dialog(title="Inference is not available for hand-crafted extraction", text="Hand-crafted feature extraction is used only as a baseline.", info="Please train your model using the end-to-ed extraction method in order to make inference available.")
            app.radioButton.setChecked(True)
            app.radioButton_2.setChecked(False)
            return
        if [f for f in os.listdir("model1") if not f.startswith('.')] == []:
            app.open_alert_dialog(title="Missing model for Inference", text="There is no machine learning model to be loaded.", info="Please use the training mode to train a model before inference.")
            app.radioButton.setChecked(True)
            app.radioButton_2.setChecked(False)
            return
        app.pushButtonRecord.setEnabled(True)
        app.pushButtonInfPlay.setEnabled(True)
        app.progressBar.setEnabled(True)
        app.progressBar_2.setEnabled(True)
        app.progressBar_3.setEnabled(True)
        app.progressBar_4.setEnabled(True)
        app.label_2.setEnabled(True)
        app.label_3.setEnabled(True)
        app.label_4.setEnabled(True)
        app.label_5.setEnabled(True)
        app.groupBox_3.setEnabled(True)
        app.groupBox_5.setEnabled(True)
        app.groupBox_2.setEnabled(False)
        app.horizontalSlider.setStyleSheet(SLYDER_DISABLED)
        app.horizontalSlider.setEnabled(False)
        app.horizontalSlider_2.setStyleSheet(SLYDER_DISABLED)
        app.horizontalSlider_2.setEnabled(False)
        app.horizontalSlider_ooda.setStyleSheet(SLYDER_DISABLED)
        app.horizontalSlider_ooda.setEnabled(False)
        ses, ser_inference_model, files = init_inference_model(app.radioButton_3.isChecked(), app.lineEdit_2.text())
        if ser_inference_model == None:
            app.open_alert_dialog(title="Missing Inference Files Alert", text="We could no find any files to classify in the stated folder.", info="You can continue the inference process by using the online model.")
            app.pushButton.setEnabled(False)
            app.pushButtonInfPlay.setEnabled(False)
        ses_online, ses_online_model = init_online_model()
    elif ses != 0 and app.radioButton.isChecked():
        app.pushButton.setEnabled(True)
        app.groupBox_2.setEnabled(True)
        app.pushButtonRecord.setEnabled(False)
        app.pushButtonInfPlay.setEnabled(False)
        app.pushButtonStopRecord.setEnabled(False)
        app.progressBar.setValue(0)
        app.progressBar.setEnabled(False)
        app.progressBar_2.setValue(0)
        app.progressBar_2.setEnabled(False)
        app.progressBar_3.setValue(0)
        app.progressBar_3.setEnabled(False)
        app.progressBar_4.setValue(0)
        app.progressBar_4.setEnabled(False)
        app.label_2.setEnabled(False)
        app.label_3.setEnabled(False)
        app.label_4.setEnabled(False)
        app.label_5.setEnabled(False)
        app.groupBox_3.setEnabled(False)
        app.groupBox_5.setEnabled(False)
        app.horizontalSlider.setStyleSheet(SLYDER_ENABLED)
        app.horizontalSlider.setEnabled(True)
        app.horizontalSlider_2.setStyleSheet(SLYDER_ENABLED)
        app.horizontalSlider_2.setEnabled(True)
        change_horizontal_ooda(app)
        close_inference_model(ses)

thread_1 = 1
def on_start_button_clicked(app):
    global thread_1
    if app.radioButton.isChecked():  # training
        app.pushButton.setEnabled(False)
        app.pushButtonStop.setEnabled(True)
        app.radioButton_2.setEnabled(False)
        app.refresh_label_7()
        app.refresh_graphics_view()
        thread_1 = Train_App(app)
        thread_1.print_accuracy_signal.connect(app.print_accuracy_graph)
        thread_1.print_stats.connect(app.print_stats_model)
        thread_1.print_matrix.connect(app.print_accuracy_matrix)
        thread_1.print_epoch.connect(app.print_label_19)
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
      if not mr.check_device_availability():
          return
      recorder_thread = Record_App(app, mr)
      recorder_thread.print_recording_signal.connect(app.print_recording_graph)
      recorder_thread.start()
      app.pushButtonStopRecord.setEnabled(True)
      app.pushButton.setEnabled(False)
      app.pushButtonInfPlay.setEnabled(False)
      app.pushButtonPlay.setEnabled(False)

def on_buttonStopRecord_clicked(app):
        import librosa
        import pyaudio
        global ses,  mr, ses_online, ses_online_model
        mr.close()
        vals = []
        if np.array(mr.get_frames()).shape[0] > 30:
            app.pushButtonPlay.setEnabled(True)
            mr.save_to_wav()        
            frames, _ = librosa.load("output.wav", 16000)    
            vals = online(ses_online, ses_online_model, frames, 44100) * 100
        else:
            app.pushButtonPlay.setEnabled(False)
            vals = [0 for _ in range(4)]
        app.progressBar.setValue(vals[0])
        app.progressBar_2.setValue(vals[1])
        app.progressBar_3.setValue(vals[2])
        app.progressBar_4.setValue(vals[3])
        print_in_label_7(app, str(list(map('{:.8f}'.format, vals))))
        app.pushButton.setEnabled(True)
        app.pushButtonRecord.setEnabled(True)
        app.pushButtonStopRecord.setEnabled(False)
        app.pushButtonInfPlay.setEnabled(True)
        app.pushButtonPlay.setEnabled(True)

nr = 0
def print_in_label_7(app, str):
    _translate = QtCore.QCoreApplication.translate
    global nr
    nr += 1
    if nr >= 24:
        txt = app.label_7.text().split("<html>")
        txt = "<html>".join(txt[1:25])
        app.label_7.setText(_translate(
            "MainWindow", txt + "<html><head/><body><span style=\" font-weight:600; color:#55ff7f;\">" + str + "</span></body></html>"))
    else:
        app.label_7.setText(_translate("MainWindow", app.label_7.text(
        ) + "<html><head/><body><span style=\" font-weight:600; color:#55ff7f;\">" + str + "</span></body></html>"))
import time

play_th = 1
def play_recording(app, file="output.wav"):
        global play_th
        play_th = Play_App(app, file)
        play_th.start()

class Train_App(QtCore.QThread):
    print_accuracy_signal = QtCore.pyqtSignal(float)
    print_stats = QtCore.pyqtSignal(str)
    print_matrix = QtCore.pyqtSignal(object)
    print_epoch = QtCore.pyqtSignal(str)
    stopFlag = False
    def __init__(self, app_rnning, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.app_rnning = app_rnning
    
    def run(self):
        print("thread")
        main(self, int(self.app_rnning.lineEdit.text()), float(self.app_rnning.horizontalSlider_2.value()) / 10,  float(self.app_rnning.horizontalSlider.value()) / 10, float(self.app_rnning.doubleSpinBox.value()) ,map_config[self.app_rnning.comboBox.currentText()], self.app_rnning.radioButton_3.isChecked())
        self.app_rnning.pushButton.setEnabled(True)
        self.app_rnning.pushButtonStop.setEnabled(False)
        self.app_rnning.radioButton_2.setEnabled(True)


class Record_App(QtCore.QThread):
    print_recording_signal = QtCore.pyqtSignal(object)

    def __init__(self, app_rnning, mr, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.app_rnning = app_rnning
        self.mr = mr
        self.app_rnning.pushButtonRecord.setEnabled(False)

    def run(self):
        self.mr.start(self)


class Play_App(QtCore.QThread):
    print_recording_signal = QtCore.pyqtSignal(object)

    def __init__(self, app_rnning, file, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.app_rnning = app_rnning
        self.file = file
        self.pushButtonInfPlayState = self.app_rnning.pushButtonPlay.isEnabled()
        self.app_rnning.pushButtonInfPlay.setEnabled(False)
        self.app_rnning.pushButtonPlay.setEnabled(False)

    def run(self):
        import wave
        import pyaudio
        chunk = 128
        f = wave.open(self.file, "rb")
        p = pyaudio.PyAudio()
        try:
                print(p.get_default_output_device_info())
        except IOError:
                print("\n\n No out device found. \n\n")
                self.app_rnning.open_alert_dialog(title="Missing output device alert", text="We could not identify any audio output device.", info="Please try and reconnec the device and restart the app.")
                return

        stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                        channels=f.getnchannels(),
                        rate= 48000,                        
                        output=True)
        data = f.readframes(chunk)
        while data:
            stream.write(data)
            data = f.readframes(chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.app_rnning.pushButtonInfPlay.setEnabled(True)
        self.app_rnning.pushButtonPlay.setEnabled(self.pushButtonInfPlayState)
