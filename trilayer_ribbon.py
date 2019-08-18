import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import os
from PyQt5.QtWidgets import QApplication
from Hamiltonian import Hamiltonian
from Lattice import lattice
from Plot import Plot
import matplotlib.pyplot as plt
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('Interface_trilayer_ribbon.ui', self)
        MainWindow.setObjectName(self, "Trilayer Ribbon")

        list1 = ['ZZ', 'AC']
        self.SelectEdge.clear()
        self.SelectEdge.addItems(list1)

        list3 = ['ABA', 'ABB', 'ABC']
        self.SelectStack.clear()
        self.SelectStack.addItems(list3)

        list2 = ['In-plane y', 'In-plane x', 'Perpendicular']
        self.SelectBBil.clear()
        self.SelectBBil.addItems(list2)

        self.CalcBands.clicked.connect(self.calc_bands)

        self.EdgeStates.clicked.connect(self.calc_edge)

        self.DOS.clicked.connect(self.dos)
        self.LDOS.clicked.connect(self.ldos)
        self.current.clicked.connect(self.currentfunc)


        self.show()

    def dos(self):

        Lat = lattice('Ribbon', self.SelectEdge.currentText() + ' ' + self.SelectStack.currentText(),
                      self.NUnit.value())

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        H.build_Hamiltonian()

        dos, ev = H.dos(self.numberofKop.value(), emin = -self.energy.value(), emax = self.energy.value(), kxmin=-np.pi, kxmax=np.pi, kymin=-np.pi, kymax=np.pi)
        plt.plot(dos[1][:-1], dos[0])
        plt.ylabel('Density of states')
        plt.xlabel('Energy')
        plt.show()

    def ldos(self):

        Lat = lattice('Ribbon', self.SelectEdge.currentText() + ' ' + self.SelectStack.currentText(),
                      self.NUnit.value())

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        H.build_Hamiltonian()
        ldos = H.expec_operators(es=np.linspace(-self.energy.value(), self.energy.value(), 100), delta=self.delta.value(), nk=self.numberofKop.value(), op=None)

        X, E = np.meshgrid(np.arange(0, len(Lat.orb[:, 1]),1), np.linspace(-self.energy.value(), self.energy.value(), 100))
        plt.contourf(X, E, ldos)
        cbar = plt.colorbar()
        cbar.set_label('Local density of states')
        plt.ylabel('Energy')
        plt.xlabel('Lattice sites')
        plt.show()

    def currentfunc(self):

        Lat = lattice('Ribbon', self.SelectEdge.currentText() + ' ' + self.SelectStack.currentText(),
                      self.NUnit.value())

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        H.build_Hamiltonian()
        current = H.expec_operators(es=np.linspace(-self.energy.value(), self.energy.value(), 100),
                                 delta=self.delta.value(), nk=self.numberofKop.value(), op="current")

        X, E = np.meshgrid(np.arange(0, len(Lat.orb[:, 1]),1), np.linspace(-self.energy.value(), self.energy.value(), 100))
        plt.contourf(X, E, current)
        cbar = plt.colorbar()
        cbar.set_label('Current')
        plt.ylabel('Energy')
        plt.xlabel('Lattice sites')
        plt.show()

    def calc_bands(self):

        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value())

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.show_rib()

    def calc_edge(self):

        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value())

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())
        if self.Magneticfield.value() != 0:
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.calc_edge()

    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())