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
        uic.loadUi('Interface_twistedbilayer_ribbon.ui', self)
        TwistedBilayer.setObjectName(self, "Twisted Bilayer Ribbon")



        # List of possible magnetic fields


        list4 = ['In-plane y', 'In-plane x', 'In-plane angle', 'Perpendicular']
        self.SelectBBil.clear()
        self.SelectBBil.addItems(list4)

        self.show()
        
        # Connect buttons

        self.CalcBands.clicked.connect(self.calc_bands)
        self.EdgeStates.clicked.connect(self.calc_edge)

    def calc_bands(self):
        
        # Calculate bandstructure from chosen lattice and Hamiltonian

        Lat = lattice('Ribbon', 'Twisted Bilayer', self.NUnit.value(), self.TwistAngle.value())

        H = Hamiltonian(Lat, self.Hoppingt.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())
        P = Plot(H)
        P.show_rib()

    def calc_edge(self):
        
        # Calculate bandstructure from chosen lattice and Hamiltonian and highlight edge states

        Lat = lattice('Ribbon', 'Twisted Bilayer', self.NUnit.value(), self.TwistAngle.value())

        H = Hamiltonian(Lat, self.Hoppingt.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())
        P = Plot(H)
        P.calc_edge()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = TwistedBilayer()
    sys.exit(app.exec_())
