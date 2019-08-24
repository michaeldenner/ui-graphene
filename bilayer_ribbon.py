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
        uic.loadUi('Interface_bilayer_ribbon.ui', self)
        MainWindow.setObjectName(self, "Bilayer Ribbon")
        
        # List of possible edge terminations

        list1 = ['ZZ', 'AC']
        self.SelectEdge.clear()
        self.SelectEdge.addItems(list1)
        
        # List of possible stacking options

        list3 = ['AB', 'AA']
        self.SelectStack.clear()
        self.SelectStack.addItems(list3)
        
        # List of possible magnetic field configurations

        list2 = ['In-plane y', 'In-plane x', 'Perpendicular']
        self.SelectBBil.clear()
        self.SelectBBil.addItems(list2)
        
        # Connect Buttons

        self.CalcBands.clicked.connect(self.calc_bands)
        self.CalcEdgeU.clicked.connect(self.upper_edge)
        self.CalcEdgeL.clicked.connect(self.lower_edge)
        self.EdgeStates.clicked.connect(self.calc_edge)
        self.LayerEdges.clicked.connect(self.calc_edge_pol)
        self.DOS.clicked.connect(self.dos)
        self.LDOS.clicked.connect(self.ldos)
        self.current.clicked.connect(self.currentfunc)

        self.show()

    def dos(self):
        
        # Calculate density of states from chosen lattice and Hamiltonian

        Lat = lattice('Ribbon', self.SelectEdge.currentText() + ' ' + self.SelectStack.currentText(),
                      self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value()) # Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        H.build_Hamiltonian()
        #Calculate dos
        dos, ev = H.dos(self.numberofKop.value(), emin = -self.energy.value(), emax = self.energy.value(), kxmin=-np.pi, kxmax=np.pi, kymin=-np.pi, kymax=np.pi)
        plt.plot(dos[1][:-1], dos[0])
        plt.ylabel('Density of states')
        plt.xlabel('Energy')
        plt.show()

    def ldos(self):
        
        # Calculate local density of states from chosen lattice and Hamiltonian

        Lat = lattice('Ribbon', self.SelectEdge.currentText() + ' ' + self.SelectStack.currentText(),
                      self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        H.build_Hamiltonian()
        # Calculate ldos
        ldos = H.expec_operators(es=np.linspace(-self.energy.value(), self.energy.value(), 100), delta=self.delta.value(), nk=self.numberofKop.value(), op=None)

        X, E = np.meshgrid(np.arange(0, len(Lat.orb[:, 1]),1), np.linspace(-self.energy.value(), self.energy.value(), 100))
        plt.contourf(X, E, ldos)
        cbar = plt.colorbar()
        cbar.set_label('Local density of states')
        plt.ylabel('Energy')
        plt.xlabel('Lattice sites')
        plt.show()

    def currentfunc(self):
        
        # Calculate current from chosen lattice and Hamiltonian

        Lat = lattice('Ribbon', self.SelectEdge.currentText() + ' ' + self.SelectStack.currentText(),
                      self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        H.build_Hamiltonian()
        # Calculate current
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
        
        # Calculate bandstructure of finite system

        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.show_rib()

    def calc_edge(self):
        
        # Calculate bandstructure and highlight edge states

        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.calc_edge()

    def lower_edge(self):
        
        # Calculate bandstructure and highlight edge states

        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.lower_edge()

    def upper_edge(self):
        
        # Calculate bandstructure and highlight edge states

        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0:# Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.upper_edge()

    def calc_edge_pol(self):
        
        # Calculate bandstructure and highlight edge states


        Lat = lattice('Ribbon', self.SelectEdge.currentText()+' '+self.SelectStack.currentText(), self.NUnit.value()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt.value(), self.Hoppingtprime.value())# Create Hamiltonian
        if self.Magneticfield.value() != 0: # Add magnetic field
            H.add_magnetic_field(self.Magneticfield.value(), self.SelectBBil.currentText())
        if self.OnsiteV.value() != 0:# Add lattice imbalance
            H.add_lattice_imbalance(self.OnsiteV.value())
        P = Plot(H)
        P.calc_layer_pol()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
