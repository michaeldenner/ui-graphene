import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import os
from PyQt5.QtWidgets import QApplication
from Hamiltonian import Hamiltonian
from Lattice import lattice
from Plot import Plot
from invariants import topology
import numpy as np
import matplotlib.pyplot as plt


class Monolayer(QtWidgets.QMainWindow):
    def __init__(self):
        super(Monolayer, self).__init__()
        uic.loadUi('Interface_monolayer_sheet.ui', self)
        Monolayer.setObjectName(self, "Monolayer Sheet")

        # Connect buttons


        self.PlotSummary.clicked.connect(self.plot_summary)
        self.Berry.clicked.connect(self.berry)
        self.Chern.clicked.connect(self.chern)
        self.LocChernK.clicked.connect(self.locchernK)
        self.LocChernKprime.clicked.connect(self.locchernKprime)
        self.DOS.clicked.connect(self.dos)


        self.show()

    def plot_summary(self):
        
        # Generate summary plot from chosen lattice and Hamiltonian
        
        K = -(4 * np.pi) / (3 * np.sqrt(3))

        if self.Haldane.value() != 0 or self.AntiHaldane.value() != 0:
            Lat = lattice('Sheet', 'Haldane') # Get correct lattice
        else:

            Lat = lattice('Sheet', 'Monolayer')


        if self.AntiHaldane.value() != 0:
            H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.AntiHaldane.value())
            H.add_antiHaldane() # If selected, add complex hopping
        else:
            H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.Haldane.value()) # If selected, add complex hopping
        if self.Magneticfield2.value() != 0: # Add magnetic field
            H.add_magnetic_field(self.Magneticfield2.value(), 'Perpendicular')
        if self.OnsiteV2.value() != 0: # Add sublattice imbalance
            H.add_lattice_imbalance(self.OnsiteV2.value())
        
        # Generate summary plot
        P = Plot(H)
        P.summary(self.Nk.value(), K, 0, self.xIndex.value(), self.yIndex.value(), self.dxIndex.value(),
                  self.dyIndex.value())
        plt.show()

    def dos(self):
        
        # Calculate density of states from chosen lattice and Hamiltonian
        
        K = -(4 * np.pi) / (3 * np.sqrt(3))

        if self.Haldane.value() != 0 or self.AntiHaldane.value() != 0:
            Lat = lattice('Sheet', 'Haldane')
        else:

            Lat = lattice('Sheet', 'Monolayer')

        if self.AntiHaldane.value() != 0:
            H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.AntiHaldane.value())
            H.add_antiHaldane()
        else:
            H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.Haldane.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), 'Perpendicular')
        if self.OnsiteV2.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV2.value())
        H.build_Hamiltonian()

        dos, ev = H.dos(self.numberofKop.value(), emin = -self.energy.value(), emax = self.energy.value(), kxmin=-np.pi, kxmax=np.pi, kymin=-np.pi, kymax=np.pi)
        plt.plot(dos[1][:-1], dos[0])
        plt.ylabel('Density of states')
        plt.xlabel('Energy')
        plt.show()




    def berry(self):
        
        # Calculate Berry phase around Dirac cones

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        if self.Haldane.value() != 0:
            Lat = lattice('Sheet', 'Haldane')
        else:

            Lat = lattice('Sheet', 'Monolayer')

        H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.Haldane.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), 'Perpendicular')
        if self.OnsiteV2.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV2.value())

        i = topology(H)

        k1 = i.berry_curvature(np.array([-(4 * np.pi) / (3 * np.sqrt(3)), 0]))
        k2 = i.berry_curvature(np.array([(4 * np.pi) / (6 * np.sqrt(3)), (4 * np.pi) / (3 * 2)]))
        k3 = i.berry_curvature(np.array([(4 * np.pi) / (6 * np.sqrt(3)), -(4 * np.pi) / (3 * 2)]))
        k4 = i.berry_curvature(np.array([(4 * np.pi) / (3 * np.sqrt(3)), 0]))
        k5 = i.berry_curvature(np.array([-(4 * np.pi) / (6 * np.sqrt(3)), (4 * np.pi) / (3 * 2)]))
        k6 = i.berry_curvature(np.array([-(4 * np.pi) / (6 * np.sqrt(3)), -(4 * np.pi) / (3 * 2)]))
        G = i.berry_curvature(np.array([0, 0]))

        msgBox = QtWidgets.QMessageBox()
        s = "K-point: {:.4f} \n K'-point: {:.4f} \n Gamma-point: {:.4f}".format(k1, k4, G)
        msgBox.setText("Berry phase")
        msgBox.setInformativeText(s)
        msgBox.exec_()

    def chern(self):
        
        # Calculate chern number from chosen lattice and Hamiltonian

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        if self.Haldane.value() != 0:
            Lat = lattice('Sheet', 'Haldane')
        else:

            Lat = lattice('Sheet', 'Monolayer')

        H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.Haldane.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), 'Perpendicular')
        if self.OnsiteV2.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV2.value())

        i = topology(H)
        ch = i.chern_int(0.01) # calculate chern number
        s = "{:.4f}".format(ch)

        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Chern number:")
        msgBox.setInformativeText(s)
        msgBox.exec_()

    

    def locchernK(self):
        
        # Calculate local chern number at K valley from chosen lattice and Hamiltonian

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        if self.Haldane.value() != 0:
            Lat = lattice('Sheet', 'Haldane')
        else:

            Lat = lattice('Sheet', 'Monolayer')

        H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.Haldane.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), 'Perpendicular')
        if self.OnsiteV2.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV2.value())

        i = topology(H)

        
        ch = i.chern_valley("K", 0.01) # calculate chern number
        s = "{:.4f}".format(ch)

        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Local Chern number:")
        msgBox.setInformativeText(s)
        msgBox.exec_()

    def locchernKprime(self):
        
        # Calculate local chern number at K' valley from chosen lattice and Hamiltonian

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        if self.Haldane.value() != 0:
            Lat = lattice('Sheet', 'Haldane')
        else:

            Lat = lattice('Sheet', 'Monolayer')

        H = Hamiltonian(Lat, self.Hoppingt2.value(), 0, 0, self.Haldane.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), 'Perpendicular')
        if self.OnsiteV2.value() != 0:
            H.add_lattice_imbalance(self.OnsiteV2.value())

        i = topology(H)

        ch = i.chern_valley("K'", 0.01) # calculate chern number
        s = "{:.4f}".format(ch)

        

        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Local Chern number:")
        msgBox.setInformativeText(s)
        msgBox.exec_()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Monolayer()
    sys.exit(app.exec_())
