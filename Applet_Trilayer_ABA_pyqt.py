import numpy as np
import sys

from matplotlib import animation
from matplotlib.contour import QuadContourSet
from matplotlib.widgets import Slider
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import HMatrix as Ham
from numpy.linalg import eigvalsh, eigh
import matplotlib.patches as patches
import matplotlib.image as mpimg
from Hamiltonian import Hamiltonian
from Lattice import lattice
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
import plotpyqt as pqt

def path(N1, N2, N3, phi=0, t6=0, t7=0, t=1,a=1):
    kx = []
    ky = []
    #Correction for shift of Dirac points
    shift_y = -phi / 2 + np.sqrt((t6 * t7) / ((1.5 * a * t)**2) + (phi * phi) / 4)
    # First path from Gamma to K
    for i in range(N1):
        kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) * (i / (N1 - 1)))
        ky.append(shift_y * (i / (N1 - 1)))
    """
    # Path from K to M
    for j in range(N2):
        kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) + (
                    (-(np.pi * np.sqrt(3)) / 3 + ((4 * np.pi) / (3 * np.sqrt(3)))) * (j / (N2 - 1))))
        ky.append(shift_y + ((np.pi / 3) - shift_y) * (j / (N2 - 1)))
    # Path from M to Gamma
    for j in range(N3):
        kx.append(-(np.pi * np.sqrt(3)) / 3 * ((N3 - j - 1) / (N3 - 1)))
        ky.append(np.pi / 3 * ((N3 - j - 1) / (N3 - 1)))
    """
    # from K to K'
    for j in range(N2):
        kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) + (
                    (-((2*np.pi ) / (3 * np.sqrt(3))) + ((4 * np.pi) / (3 * np.sqrt(3)))) * (j / (N2 - 1))))
        ky.append(shift_y + (((2*np.pi) / 3) - shift_y) * (j / (N2 - 1)))

    # from K' to Gamma
    for j in range(N3):
        kx.append(-((2*np.pi) / (3 * np.sqrt(3))) * ((N3 - j - 1) / (N3 - 1)))
        ky.append((2*np.pi) / 3 * ((N3 - j - 1) / (N3 - 1)))


    return kx, ky

def evals1(phi,theta):
    N = 150
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    KX, KY = np.meshgrid(kx, ky)
    H.add_magnetic_field(phi, 'In-plane angle', theta)


    eigenvals = H.energy(N)


    #H = Ham.Bilayer_origin_A2_field_angle(KX, KY, N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7*t6, phi, theta, a, 0)
    #eigenvals = eigvalsh(H)
    return eigenvals[:, :, 2]

def evals2(phi,theta):
    K = -(4 * np.pi) / (3 * np.sqrt(3))
    N = 150
    dkx = np.linspace(K - 0.4, K + 0.4, N + 1)  # (-0.6, 0.6, N+1)
    dky = np.linspace(-0.4, 0.4, N + 1)  # (K-0.6, K+0.6, N+1)
    DKX, DKY = np.meshgrid(dkx, dky)
    H.add_magnetic_field(phi, 'In-plane angle', theta)


    #H = Ham.Bilayer_origin_A2_field_angle(DKX, DKY, N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7*t6, phi, theta, a, 0)
    eigenvals = H.energy(N, K - 0.4, K + 0.4, 0 - 0.4, 0 + 0.4)
    return eigenvals

def eival(phi, theta):
    N1 = 150
    N2 = 150
    N3 = 150

    kx, ky = path(N1, N2, N3)
    eigenvals1 = []
    eigenvals2 = []
    eigenvals3 = []
    eigenvals4 = []
    eigenvals5 = []
    eigenvals6 = []
    H.add_magnetic_field(phi, 'In-plane angle', theta)

    for i in range(len(kx)):
        eigval = eigvalsh(
            H.get_Hamiltonian(kx[i], ky[i]))
        eigenvals1.append(eigval[0])
        eigenvals2.append(eigval[1])
        eigenvals3.append(eigval[2])
        eigenvals4.append(eigval[3])
        eigenvals5.append(eigval[4])
        eigenvals6.append(eigval[5])

    return eigenvals1,eigenvals2,eigenvals3,eigenvals4,eigenvals5,eigenvals6






def compute_and_plot_1(ax, alpha, theta):
    #Calculate grid values
    N = 150
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    KX, KY = np.meshgrid(kx, ky)


    CS1 = QuadContourSet(ax, KY, KX, evals1(alpha, theta),levels=[-4,-3,-2, -1, -0.5, -0.25, 0], cmap=plt.get_cmap('Reds'), filled=True)

    #pyl.clabel(CS1, inline=1, fontsize=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$k_ya$", fontsize=5)
    ax.set_ylabel("$k_xa$", fontsize=5)
    ax.set_title('Constant energy lines in 1.BZ', fontsize=5)
    rect = patches.Rectangle((-2.9, -0.5), 1.0, 1.0, linewidth=0.5, edgecolor='black', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    #cbar = pyl.colorbar(CS1)
    #ax.tick_params(labelsize=5)



def compute_and_plot_2(ax, alpha, theta):
    #Calculate grid values
    N = 150
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    KX, KY = np.meshgrid(kx, ky)
    E = evals2(alpha, theta)

    CS2 = QuadContourSet(ax, KY, KX, E[:, :, 2],levels=[-1, -0.5, -0.25, -0.1, -0.01, 0], cmap=plt.get_cmap('Reds'), linewidth=1)
    plt.clabel(CS2, inline=1, fontsize=5)
    ax.set_xlabel("$k_ya$", fontsize=5)
    ax.set_ylabel("$k_xa$", fontsize=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Constant energy lines around K point', fontsize=5)


def compute_and_plot_3(ax, phi, theta):
    eigenvals1, eigenvals2, eigenvals3, eigenvals4,eigenvals5,eigenvals6 = eival(phi, theta)
    N1 = 150
    N2 = 150
    N3 = 150

    axis = np.arange(0, N1 + N2 + N3, 1)



    ax.plot(axis, eigenvals1, color="b", linewidth=1)
    ax.plot(axis, eigenvals2, color="y", linewidth=1)
    ax.plot(axis, eigenvals3, color="g", linewidth=1)
    ax.plot(axis, eigenvals4, color="r", linewidth=1)
    ax.plot(axis, eigenvals5, linewidth=1)
    ax.plot(axis, eigenvals6, linewidth=1)
    ax.set_yticks([])
    ax.set_xticks([0,150,300,450])
    ax.set_xticklabels(["$\Gamma$", "$K$", "$K'$", "$\Gamma$"], fontsize = 5)
    ax.set_title('Energy spectrum', fontsize=5)
    ax.set_ylabel("$\epsilon_k$", fontsize=5)

def compute_and_plot_4(ax, alpha, theta):
    E = evals2(alpha, theta)
    x = 75
    ax.plot(H.axis2[:, 0], E[x, :, 0], linewidth=1)
    ax.plot(H.axis2[:, 0], E[x, :, 1], linewidth=1)
    ax.plot(H.axis2[:, 0], E[x, :, 2], linewidth=1)
    ax.plot(H.axis2[:, 0], E[x, :, 3], linewidth=1)
    ax.plot(H.axis2[:, 0], E[x, :, 4], linewidth=1)
    ax.plot(H.axis2[:, 0], E[x, :, 5], linewidth=1)
    ax.set_xticks([0])
    ax.set_yticks([])
    ax.set_xticklabels(["$K$"], fontsize=5)
    ax.set_xlabel("$k_ya$", fontsize=5)
    ax.set_title('Plot along ky', fontsize=5)
    ax.set_ylabel("$\epsilon_{k}$", fontsize=5)








alpha = 0
theta = 0
t7 = 1

Lat = lattice('Sheet', 'Trilayer ABA')

H = Hamiltonian(Lat, 3, 0.3)

H.build_Hamiltonian()

def run():

    def funfig(obj):  # dummy function
        #plt.close()

        # Plot
        fig = plt.figure(obj.figure.number)
        fig.clear()

        # pyl.title('Simplest default with labels')
        ax1 = fig.add_subplot(221)

        compute_and_plot_1(ax1, obj.get_slider("B"), obj.get_slider("Angle"))
        ax2 = fig.add_subplot(222)
        compute_and_plot_2(ax2, obj.get_slider("B"), obj.get_slider("Angle"))

        ax3 = fig.add_subplot(223)
        compute_and_plot_3(ax3, obj.get_slider("B"), obj.get_slider("Angle"))

        # ax3.set_xticks(["$\Gamma$", "$K$", "$K'$", "$\Gamma$"])

        ax4 = fig.add_subplot(224)
        compute_and_plot_4(ax4, obj.get_slider("B"), obj.get_slider("Angle"))
        plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
        plt.tight_layout()





        return fig  # return figure

    app, main = pqt.get_interface(funfig)  # get the interface
    Bs = np.linspace(0.0, 2*np.pi, 100)  # field strengths
    thetas = np.linspace(0.0, np.pi/2, 50)  # angles
    main.add_slider(label="Magnetic field", key="B", vs=Bs)  # initialize the slider
    main.add_slider(label="Angle", key="Angle", vs=thetas)  # initialize the slider
    # initialize the combobox

    main.plot()
    main.show()
    #sys.exit(app.exec_())

