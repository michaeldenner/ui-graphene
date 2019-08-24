import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication
from Hamiltonian import Hamiltonian
from Lattice import lattice
from Plot import Plot
from invariants import topology
import numpy as np
import Applet_Trilayer_ABA_pyqt as ap_ABA
import Applet_Trilayer_ABB as ap_ABB
import Applet_Trilayer_ABC as ap_ABC
import matplotlib.pyplot as plt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('Interface_trilayer_sheet.ui', self)
        MainWindow.setObjectName(self, "Trilayer Sheet")

        # List of possible stacking geometries

        list2 = ['Trilayer ABA', 'Trilayer ABB', 'Trilayer ABC']
        self.SelectStack.clear()
        self.SelectStack.addItems(list2)

        # List of possible magnetic fields

        list4 = ['In-plane y', 'In-plane x', 'In-plane angle', 'Perpendicular', 'Artificial']
        self.SelectBBil.clear()
        self.SelectBBil.addItems(list4)

        self.show()
        
        # Connect buttons

        self.PlotSummary.clicked.connect(self.plot_summary)
        self.Phases.clicked.connect(self.phases)
        self.Berry.clicked.connect(self.berry)
        self.Chern.clicked.connect(self.chern)
        self.LocChernK.clicked.connect(self.locchernK)
        self.LocChernKprime.clicked.connect(self.locchernKprime)
        self.DOS.clicked.connect(self.dos)


    def plot_summary(self):
        
        # Generate summary plot from chosen lattice and Hamiltonian
        
        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', self.SelectStack.currentText()) # Create lattice

        H = Hamiltonian(Lat, self.Hoppingt2.value(), self.Hoppingtprime2.value()) # Create Hamiltonian
        if self.Magneticfield2.value() != 0: # Add magnetic field
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())

        if self.OnsiteV_2.value() != 0: # Add layer polarization
            H.add_sublattice_imbalance(self.OnsiteV_2.value())
        P = Plot(H)
        P.summary(self.Nk.value(), K, 0, self.xIndex.value(), self.yIndex.value(), self.dxIndex.value(), self.dyIndex.value())

    def dos(self):
        
        # calculate density of states from chosen lattice and Hamiltonian
        
        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', self.SelectStack.currentText())

        H = Hamiltonian(Lat, self.Hoppingt2.value(), self.Hoppingtprime2.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())

        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())
        H.build_Hamiltonian()
        
        # calculate dos

        dos, ev = H.dos(self.numberofKop.value(), emin=-self.energy.value(), emax=self.energy.value(), kxmin=-np.pi,
                        kxmax=np.pi, kymin=-np.pi, kymax=np.pi)
        plt.plot(dos[1][:-1], dos[0])
        plt.ylabel('Density of states')
        plt.xlabel('Energy')
        plt.show()


    def phases(self):
        # run applets
        if self.SelectStack.currentText() == 'Trilayer ABA':
            ap_ABA.run()

        elif self.SelectStack.currentText() == 'Trilayer ABB':
            ap_ABB.run()

        else:
            ap_ABC.run()



    def berry(self):
        # calculate Berry phases
        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', self.SelectStack.currentText())

        H = Hamiltonian(Lat, self.Hoppingt2.value(), self.Hoppingtprime2.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())
        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())

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
        
        # Calculate chern number

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', self.SelectStack.currentText())

        H = Hamiltonian(Lat, self.Hoppingt2.value(), self.Hoppingtprime2.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())
        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())

        i = topology(H)
        ch = i.chern_int(0.01)
        s = "{:.4f}".format(ch)


        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Chern number:")
        msgBox.setInformativeText(s)
        msgBox.exec_()

        #print("Calculating Chern number ....")
        #print()

    def locchernK(self):
        
        # Calculate local chern number in K valley

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', self.SelectStack.currentText())

        H = Hamiltonian(Lat, self.Hoppingt2.value(), self.Hoppingtprime2.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())

        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())

        i = topology(H)

        #print("Local Chern number:", i.chern_valley('K', 0.01))
        ch = i.chern_valley("K", 0.01)
        s = "{:.4f}".format(ch)


        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Local Chern number:")
        msgBox.setInformativeText(s)
        msgBox.exec_()

    def locchernKprime(self):
        
        # Calculate local chern number in K' valley

        K = -(4 * np.pi) / (3 * np.sqrt(3))

        Lat = lattice('Sheet', self.SelectStack.currentText())

        H = Hamiltonian(Lat, self.Hoppingt2.value(), self.Hoppingtprime2.value())
        if self.Magneticfield2.value() != 0:
            H.add_magnetic_field(self.Magneticfield2.value(), self.SelectBBil.currentText(), self.BAngle.value())

        if self.OnsiteV_2.value() != 0:
            H.add_sublattice_imbalance(self.OnsiteV_2.value())

        i = topology(H)

        ch = i.chern_valley("K'", 0.01)
        s = "{:.4f}".format(ch)

        #print("Local Chern number:", i.chern_valley("K'", 0.01))

        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Local Chern number:")
        msgBox.setInformativeText(s)
        msgBox.exec_()





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
