import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import os
from PyQt5.QtWidgets import QApplication
from Hamiltonian import Hamiltonian
from Lattice import lattice
from Plot import Plot
from invariants import topology
import numpy as np
import Applet as ap
import Applet_Trilayer_ABA_pyqt as ap_ABA
import Applet_Trilayer_ABB as ap_ABB
import Applet_Trilayer_ABC as ap_ABC



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('Interface.ui', self)
        MainWindow.setObjectName(self, "Master Project")

        self.show()
        
        # Launch interface and connect buttons

        self.monolayer_ribbon.clicked.connect(lambda:os.system('python monolayer_ribbon.py'))
        self.bilayer_ribbon.clicked.connect(lambda:os.system('python bilayer_ribbon.py'))
        self.trilayer_ribbon.clicked.connect(lambda:os.system('python trilayer_ribbon.py'))
        self.twistedbilayer_ribbon.clicked.connect(lambda:os.system('python twistedbilayer_ribbon.py'))
        self.monolayer_sheet.clicked.connect(lambda:os.system('python monolayer_sheet.py'))
        self.bilayer_sheet.clicked.connect(lambda:os.system('python bilayer_sheet.py'))
        self.trilayer_sheet.clicked.connect(lambda:os.system('python trilayer_sheet.py'))
        self.twistedbilayer_sheet.clicked.connect(lambda:os.system('python twistedbilayer_sheet.py'))




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
