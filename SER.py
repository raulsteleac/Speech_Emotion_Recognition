from PyQt5 import QtCore, QtGui, QtWidgets
from graphics.UI_Class_Definition import Ui_MainWindow
from model import main, inference
from util import *

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
