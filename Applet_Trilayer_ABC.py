import numpy as np


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
from matplotlib import pyplot as pyl

def path(N1, N2, N3, phi=0, t6=0, t7=0, t=1,a=1):
    kx = []
    ky = []
    #Correction for shift of Dirac points
    shift_y = -phi / 2 + np.sqrt((t6 * t7) / ((1.5 * a * t)**2) + (phi * phi) / 4)
    # First path from Gamma to K
    for i in range(N1):
        kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) * (i / (N1 - 1)))
        ky.append(shift_y * (i / (N1 - 1)))

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

def evals1(phi,theta,t7):
    
    N = 150 # Number of k-points
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    KX, KY = np.meshgrid(kx, ky)
    H.add_magnetic_field(phi, 'In-plane angle', theta) # Add magnetic field
    
    # List of parameters
    
    a = 1
    t1 = 3
    t2 = 3
    t3 = 3
    t6 = 0.4
    t1_tilde = 3
    t2_tilde = 3
    t3_tilde = 3

    eigenvals = H.energy(N) # Get energies


    
    return eigenvals[:, :, 2]

def evals2(phi,theta,t7):
    K = -(4 * np.pi) / (3 * np.sqrt(3)) # Location of Dirac cones
    N = 150 # Number of k-points
    dkx = np.linspace(K - 0.4, K + 0.4, N + 1)
    dky = np.linspace(-0.4, 0.4, N + 1)
    DKX, DKY = np.meshgrid(dkx, dky)
    H.add_magnetic_field(phi, 'In-plane angle', theta) # Add magnetic field
    
    # List of parameters
    
    a = 1
    t1 = 3
    t2 = 3
    t3 = 3
    t6 = 0.4

    t1_tilde = 3
    t2_tilde = 3
    t3_tilde = 3

    
    eigenvals = H.energy(N, K - 0.4, K + 0.4, 0 - 0.4, 0 + 0.4)
    return eigenvals

def eival(phi, theta, t7):
    
    # List of parameters
    
    N1 = 150
    N2 = 150
    N3 = 150
    a = 1
    t1 = 3
    t2 = 3
    t3 = 3
    t6 = 0.4

    t1_tilde = 3
    t2_tilde = 3
    t3_tilde = 3
    kx, ky = path(N1, N2, N3) # Path of k-points in 1.BZ
    eigenvals1 = []
    eigenvals2 = []
    eigenvals3 = []
    eigenvals4 = []
    eigenvals5 = []
    eigenvals6 = []
    H.add_magnetic_field(phi, 'In-plane angle', theta) # Add magnetic field

    for i in range(len(kx)):
        eigval = eigvalsh(H.get_Hamiltonian(kx[i], ky[i])) # Get energies
        eigenvals1.append(eigval[0])
        eigenvals2.append(eigval[1])
        eigenvals3.append(eigval[2])
        eigenvals4.append(eigval[3])
        eigenvals5.append(eigval[4])
        eigenvals6.append(eigval[5])

    return eigenvals1,eigenvals2,eigenvals3,eigenvals4,eigenvals5,eigenvals6






def compute_and_plot_1(ax, alpha, theta, t7):
    #Calculate grid values
    N = 150
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    KX, KY = np.meshgrid(kx, ky)


    CS1 = QuadContourSet(ax, KY, KX, evals1(alpha, theta, t7),levels=[-4,-3,-2, -1, -0.5, -0.25, 0], cmap=pyl.get_cmap('Reds'), filled=True) # Plot Fermisurface


    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$k_ya$", fontsize=8)
    ax.set_ylabel("$k_xa$", fontsize=8)
    ax.set_title('Constant energy lines in 1.BZ', fontsize=6)
    rect = patches.Rectangle((-2.9, -0.5), 1.0, 1.0, linewidth=1, edgecolor='black', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)




def compute_and_plot_2(ax, alpha, theta, t7):
    #Calculate grid values
    N = 150
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    KX, KY = np.meshgrid(kx, ky)
    E = evals2(alpha, theta, t7)

    CS2 = QuadContourSet(ax, KY, KX, E[:, :, 2],levels=[-1, -0.5, -0.25, -0.1, -0.01, 0], cmap=pyl.get_cmap('Reds'))# Plot Fermisurface

    pyl.clabel(CS2, inline=1, fontsize=5)
    ax.set_xlabel("$k_ya$", fontsize=8)
    ax.set_ylabel("$k_xa$", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Constant energy lines around K point', fontsize=6)


def compute_and_plot_3(ax, phi, theta, t7):
    eigenvals1, eigenvals2, eigenvals3, eigenvals4,eigenvals5,eigenvals6 = eival(phi, theta, t7)
    N1 = 150
    N2 = 150
    N3 = 150

    axis = np.arange(0, N1 + N2 + N3, 1)

    # Plot Bandstructure along path in BZ


    ax.plot(axis, eigenvals1, color="b")
    ax.plot(axis, eigenvals2, color="y")
    ax.plot(axis, eigenvals3, color="g")
    ax.plot(axis, eigenvals4, color="r")
    ax.plot(axis, eigenvals5)
    ax.plot(axis, eigenvals6)
    ax.set_yticks([])
    ax.set_xticks([0,150,300,450])
    ax.set_xticklabels(["$\Gamma$", "$K$", "$K'$", "$\Gamma$"], fontsize = 8)
    ax.set_title('Energy spectrum', fontsize=8)
    ax.set_ylabel("$\epsilon_k$", fontsize=8)

def compute_and_plot_4(ax, alpha, theta, t7):
    E = evals2(alpha, theta, t7)
    x = 75
    
    # Plot Bandstructure along ky for x = 75

    
    ax.plot(H.axis2[:, 0], E[x, :, 0])
    ax.plot(H.axis2[:, 0], E[x, :, 1])
    ax.plot(H.axis2[:, 0], E[x, :, 2])
    ax.plot(H.axis2[:, 0], E[x, :, 3])
    ax.plot(H.axis2[:, 0], E[x, :, 4])
    ax.plot(H.axis2[:, 0], E[x, :, 5])


    ax.set_xlabel("$k_ya$")
    ax.set_title('Plot along ky', fontsize=10)
    ax.set_ylabel("$\epsilon_{k}$")






# List of parameters

alpha = 0
theta = 0
t7 = 1

# Get lattice

Lat = lattice('Sheet', 'Trilayer ABC')

# Get Hamiltonian

H = Hamiltonian(Lat, 3, 0.3)

H.build_Hamiltonian()

def run():

    def update(ax1, ax2, ax3, ax4, val):

        alpha = alpha_slider.val

        theta = theta_slider.val
        t7 = t7_slider.val
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        compute_and_plot_1(ax1, alpha, theta, t7)
        compute_and_plot_2(ax2, alpha, theta, t7)
        compute_and_plot_3(ax3, alpha, theta, t7)
        compute_and_plot_4(ax4, alpha, theta, t7)
        pyl.draw()



    # Plot
    fig = pyl.figure(figsize=(16,9))

    
    ax1 = fig.add_subplot(331)

    compute_and_plot_1(ax1, alpha, theta, t7)
    ax2 = fig.add_subplot(332)
    compute_and_plot_2(ax2, alpha, theta, t7)

    ax3 = fig.add_subplot(333)
    compute_and_plot_3(ax3, alpha, theta, t7)


    ax4 = fig.add_subplot(335)
    compute_and_plot_4(ax4, alpha, theta, t7)

    ax5 = fig.add_subplot(336)
    img = mpimg.imread('phases.png')
    ax5.imshow(img)
    ax5.axis('off')

    pyl.subplots_adjust(wspace=0.3)

    # Define slider for alpha
    axcolor = 'white'
    alpha_axis = pyl.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    alpha_slider = Slider(alpha_axis, '$\Phi$', 0, 4 * np.pi / np.sqrt(3), valinit=.0)
    theta_axis = pyl.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    theta_slider = Slider(theta_axis, 'Angle', 0, np.pi / 2, valinit=.0)
    t7_axis = pyl.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    t7_slider = Slider(t7_axis, '$t_{bb}$ (in units of $t_{aa}$)', 0.0, 1.0, valinit=0.0)

    alpha_slider.on_changed(lambda val: update(ax1, ax2, ax3, ax4, val))
    theta_slider.on_changed(lambda val: update(ax1, ax2, ax3, ax4, val))
    t7_slider.on_changed(lambda val: update(ax1, ax2, ax3, ax4, val))

    fig.canvas.set_window_title('Phase Applet')



    pyl.show()
