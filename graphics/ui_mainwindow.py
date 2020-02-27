# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from util import *
from model import main, inference
from UI_Class_Definition import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMouseTracking(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.trainGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.trainGroupBox.setMouseTracking(True)
        self.trainGroupBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.trainGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.trainGroupBox.setFlat(True)
        self.trainGroupBox.setCheckable(False)
        self.trainGroupBox.setObjectName("trainGroupBox")
        self.horizontalLayout_2.addWidget(self.trainGroupBox)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout_2.addWidget(self.line_2)
        self.inferenceGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.inferenceGroupBox.setMouseTracking(True)
        self.inferenceGroupBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.inferenceGroupBox.setTitle("Inference")
        self.inferenceGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.inferenceGroupBox.setObjectName("inferenceGroupBox")
        self.trainGroupBox.raise_()
        self.horizontalLayout_2.addWidget(self.inferenceGroupBox)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.projectTitleLabel = QtWidgets.QLabel(self.centralwidget)
        self.projectTitleLabel.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.projectTitleLabel.sizePolicy().hasHeightForWidth())
        self.projectTitleLabel.setSizePolicy(sizePolicy)
        self.projectTitleLabel.setTextFormat(QtCore.Qt.PlainText)
        self.projectTitleLabel.setScaledContents(False)
        self.projectTitleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.projectTitleLabel.setWordWrap(False)
        self.projectTitleLabel.setIndent(1)
        self.projectTitleLabel.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.projectTitleLabel.setObjectName("projectTitleLabel")
        self.verticalLayout_3.addWidget(self.projectTitleLabel)
        self.gridLayout_2.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # self.trainGroupBox.title.setStyleSheet("subcontrol-origin: margin"
        #                                        "subcontrol-position: top center"
        #                                        "padding-left: 0.1ex"
        #                                        "padding-right: 0.1ex"
        #                                        "margin-top: -0.7ex"
        #                                        "font: bold 14px"
        #                                        )

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.trainGroupBox.setTitle(_translate("MainWindow", "Train"))
        self.projectTitleLabel.setText(_translate(
            "MainWindow", "Speech Emotion Recognition in an end-to-end setting"))


class SER_GUI(QtWidgets.QMainWindow):
      default_main_window_w = 1500
      default_main_window_h = 900

      def __init__(self, w=default_main_window_w, h=default_main_window_h):
            super().__init__()

      def draw(self):
          ui = Ui_MainWindow()
          ui.setupUi(self)
          self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    file = QtCore.QFile("./BreezeStyleSheets-master/dark.qss")
    file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
    stream = QtCore.QTextStream(file)
    app.setStyleSheet(stream.readAll())
    main_Window = SER_GUI()
    main_Window.draw()
    app.exec_()
