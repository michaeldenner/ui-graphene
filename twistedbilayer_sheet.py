import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication
from Hamiltonian import Hamiltonian
from Lattice import lattice
from Plot import Plot
from invariants import topology
import numpy as np


class TwistedBilayer(QtWidgets.QMainWindow):
    def __init__(self):
        super(TwistedBilayer, self).__init__()
        uic.loadUi('Interface_twistedbilayer_sheet.ui', self)
        TwistedBilayer.setObjectName(self, "Twisted Bilayer Sheet")


        # List of possible magnetic fields



        list4 = ['In-plane y', 'In-plane x', 'In-plane angle', 'Perpendicular']
        self.SelectBBil.clear()
        self.SelectBBil.addItems(list4)

        self.show()
        
        # Connect buttons

        self.PlotSummary.clicked.connect(self.plot_summary)

    def plot_summary(self):
        
        # Generate summary plot of chosen lattice and Hamiltonian
        
        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', 'Twisted Bilayer', 1, self.TwistAngle.value())

        H = Hamiltonian(Lat, self.Hoppingt2.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())
        P = Plot(H)
        P.summary(self.Nk.value(), K, 0, 17, 75, 75, 75)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = TwistedBilayer()
    sys.exit(app.exec_())
