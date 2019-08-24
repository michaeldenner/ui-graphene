import numpy as np

from numpy.linalg import eigvalsh, eigh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors
from matplotlib.colors import Normalize
import os




class Plot:
    def __init__(self, Hamiltonian):
        self.Hamiltonian = Hamiltonian

    def Plot_Fermi_surface_s(self, ax, x, y):
        """
        Plot the 2D Fermi surface
        :param KX: meshgrid of kx values
        :param KY: meshgrid of ky values
        :param eigenvals: eigenvalues to be plotted
        :return: True
        """

        # Plot the 2D Fermi surface

        if self.Hamiltonian.lattice.layer == 1:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.BZ[:, :, 0]),
                            levels=[-2, -1, -0.5, -0.25, -0.1, 0], cmap=plt.get_cmap('Reds'))
            c2 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.BZ[:, :, 1]),
                            levels=[0, 0.1, 0.25, 0.5, 1, 2], cmap=plt.get_cmap('Blues_r'))
        elif self.Hamiltonian.lattice.layer == 2:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.BZ[:, :, 1]),
                            levels=[-2, -1, -0.5, -0.25, -0.1, 0], cmap=plt.get_cmap('Reds'))
            c2 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.BZ[:, :, 2]),
                            levels=[0, 0.1, 0.25, 0.5, 1, 2], cmap=plt.get_cmap('Blues_r'))
        else:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.BZ[:, :, 2]),
                            levels=[-2, -1, -0.5, -0.25, -0.1, 0], cmap=plt.get_cmap('Reds'))
            c2 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.BZ[:, :, 3]),
                            levels=[0, 0.1, 0.25, 0.5, 1, 2], cmap=plt.get_cmap('Blues_r'))


        plt.axhline(self.Hamiltonian.axis1[0][y], color='black')

        plt.axvline(self.Hamiltonian.axis1[0][x], color='black')

        ax.set_title('Constant energy lines in 1.BZ', fontsize=10)
        ax.set_xlabel("$k_xa$")
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(["$-\pi$", "$0$", "$\pi$"])
        ax.set_ylabel("$k_ya$")
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(["$-\pi$", "$0$", "$\pi$"])

    def Plot_along_kx_s(self, ax, y, E):
        """
        Plot the energy spectrum for a constant momentum ky along kx
        :param kx: meshgrid of kx values
        :param y: momentum to be used
        :param eigenvals: eigenvalues to be plotted
        :return: True
        """

        if self.Hamiltonian.lattice.layer == 1:
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 0], color="b")
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 1], color="y")

        elif self.Hamiltonian.lattice.layer == 2:
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 0], color="b")
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 1], color="y")
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 2], color="g")
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 3], color="r")
        else:
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 0])
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 1])
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 2])
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 3])
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 4])
            plt.plot(self.Hamiltonian.axis1[0], E[:, y, 5])

        fo = open(os.path.join('/tmp', "Bands_kx.TXT"), "w")

        for ik in range(len(self.Hamiltonian.axis1[0])):
            for ie in range(len(self.Hamiltonian.lattice.orb)):
                fo.write(str(ik) + "   " + str(E[ik, y, ie]))
                fo.write("\n")
        fo.close()

        ax.set_xlabel("$k_xa$")
        ax.set_title('Plot along kx', fontsize=10)
        ax.set_ylabel("$\epsilon_{k}$")

    def Plot_along_ky_s(self, ax, x, E):
        """
        Plot the energy spectrum for a constant momentum kx along ky
        :param ky: meshgrid of ky values
        :param x: Momentum to be used
        :param eigenvals: eigenvalues to be plotted
        :return: True
        """

        if self.Hamiltonian.lattice.layer == 1:
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 0], color="b")
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 1], color="y")

        elif self.Hamiltonian.lattice.layer == 2:

            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 0], color="b")
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 1], color="y")
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 2], color="g")
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 3], color="r")

        else:
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 0])
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 1])
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 2])
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 3])
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 4])
            plt.plot(self.Hamiltonian.axis2[:, 0], E[x, :, 5])



        ax.set_xlabel("$k_ya$")
        ax.set_title('Plot along ky', fontsize=10)
        ax.set_ylabel("$\epsilon_{k}$")

    def Plot_Fermi_surface_zoomed_s(self, ax, x, y):

        """
        Plot the 2D Fermi surface around the K point
        :param KX: meshgrid of kx values
        :param KY: meshgrid of ky values
        :param eigenvals: eigenvalues to be plotted
        """

        if self.Hamiltonian.lattice.layer == 1:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 0]),
                            levels=[-1, -0.5, -0.25, -0.1, -0.01, -0.001, 0], cmap=plt.get_cmap('Reds'))

            c2 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 1]),
                            levels=[0, 0.001, 0.01, 0.1, 0.25, 0.5, 1], cmap=plt.get_cmap('Blues_r'))

        elif self.Hamiltonian.lattice.layer == 2:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 1]),
                            levels=[-1, -0.5, -0.25, -0.1, -0.01, -0.001, 0], cmap=plt.get_cmap('Reds'))

            c2 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 2]),
                            levels=[0, 0.001, 0.01, 0.1, 0.25, 0.5, 1], cmap=plt.get_cmap('Blues_r'))

        else:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 2]),
                            levels=[-1, -0.5, -0.25, -0.1, -0.01, -0.001, 0], cmap=plt.get_cmap('Reds'))

            c2 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 3]),
                            levels=[0, 0.001, 0.01, 0.1, 0.25, 0.5, 1], cmap=plt.get_cmap('Blues_r'))

        ax.axhline(self.Hamiltonian.axis2[y][0], color='black')

        ax.axvline(self.Hamiltonian.axis1[0][x], color='black')

        ax.set_title('Constant energy lines at K point', fontsize=10)
        plt.clabel(c1, inline=1, fontsize=10, fmt="%.2f")
        plt.clabel(c2, inline=1, fontsize=10, fmt="%.2f")
        ax.set_xlabel("$k_xa$")
        ax.set_xticks([0])
        ax.set_xticklabels(["K"])
        ax.set_ylabel("$k_ya$")
        ax.set_yticks([0])
        ax.set_yticklabels(["0"])

    def Plot_Fermi_surface_zoomed_one_band_s(self, ax):

        """
        Plot the 2D Fermi surface around the K point
        :param KX: meshgrid of kx values
        :param KY: meshgrid of ky values
        :param eigenvals: eigenvalues to be plotted
        :return: True
        """

        if self.Hamiltonian.lattice.layer == 1:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 0]),
                            levels=[-2,-1, -0.5, -0.25, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 0.25, 0.5, 1,2], colors='k')
            cs1 = ax.contourf(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 0]),
                              cmap=plt.get_cmap('Reds_r'))


        elif self.Hamiltonian.lattice.layer == 2:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 1]),
                            levels=[-2, -1, -0.5,-0.25,-0.1,-0.01, -0.001, -0.0001,-0.00001, 0,0.00001, 0.0001, 0.001, 0.01,0.1,0.25, 0.5, 1 ,2], colors='k')
            cs1 = ax.contourf(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 1]),
                              cmap=plt.get_cmap('Reds_r'))
        else:
            c1 = ax.contour(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 2]),
                            levels=[-1, -0.5, -0.25, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 0.25, 0.5, 1], colors='k')
            cs1 = ax.contourf(self.Hamiltonian.axis1, self.Hamiltonian.axis2, np.transpose(self.K[:, :, 2]),
                              cmap=plt.get_cmap('Reds_r'))

        fo = open(os.path.join('/tmp', "Bands_FS.TXT"), "w")

        for ikx in range(len(self.Hamiltonian.axis1)):
            for iky in range(len(self.Hamiltonian.axis2)):
                fo.write(str(ikx) + "   " + str(iky) + "   " + str(self.K[ikx, iky, 1]))
                fo.write("\n")
        fo.close()

        cbar = plt.colorbar(cs1)
        cbar.ax.set_ylabel('Energy')
        cbar.add_lines(c1)

        # Add the contour line levels to the colorbar
        ax.set_ylabel("$k_y$", fontsize=10)
        ax.set_yticks([0])
        ax.set_yticklabels(["0"], fontsize=10)
        ax.set_xticks([-(4 * np.pi) / (3 * np.sqrt(3))])
        ax.set_xticklabels(["K"], fontsize=10)
        plt.clabel(c1, inline=1, fontsize=10, fmt="%.2f")
        ax.set_title('Constant energy lines at K point', fontsize=10)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):

            label.set_fontsize(10)



        ax.set_xlabel("$k_x$", fontsize=30)

    def path(self, N):
        """
        Generate a path of k values
        :param N: number of evaluation points
        :return: k momenta on path
        """
        kx = []
        ky = []

        # First path from Gamma to K
        for i in range(N):
            kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) * (i / (N - 1)))
            ky.append(0)

        # from K to K'
        for j in range(N):
            kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) + (
                    (-((2 * np.pi) / (3 * np.sqrt(3))) + ((4 * np.pi) / (3 * np.sqrt(3)))) * (j / (N - 1))))
            ky.append((((2 * np.pi) / 3)) * (j / (N - 1)))

        # from K' to Gamma
        for j in range(N):
            kx.append(-((2 * np.pi) / (3 * np.sqrt(3))) * ((N - j - 1) / (N - 1)))
            ky.append((2 * np.pi) / 3 * ((N - j - 1) / (N - 1)))

        return kx, ky

    def path_theta_0(self, N):
        """
        Generate a path of k values
        :param N: number of evaluation points
        :return: k momenta on path
        """
        kx = []
        ky = []

        # First path from Gamma to K'
        for i in range(N):
            kx.append(-((2 * np.pi) / (3 * np.sqrt(3))) * (i / (N - 1)))
            ky.append(((2 * np.pi) / (3)) * (i / (N - 1)))

        # from K' to K
        for j in range(N):
            kx.append(-((2 * np.pi) / (3 * np.sqrt(3))) + (
                    ((4 * np.pi) / (3 * np.sqrt(3))) * (j / (N - 1))))
            ky.append(((2 * np.pi) / 3))

        # from K' to Gamma
        for j in range(N):
            kx.append(((2 * np.pi) / (3 * np.sqrt(3))) * ((N - j - 1) / (N - 1)))
            ky.append((2 * np.pi) / 3 * ((N - j - 1) / (N - 1)))

        return kx, ky

    def path_theta_pi_6(self, N):
        """
        Generate a path of k values
        :param N: number of evaluation points
        :return: k momenta on path
        """
        kx = []
        ky = []

        # First path from Gamma to K
        for i in range(N):
            kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) * (i / (N - 1)))
            ky.append(0)

        # from K to K'
        for j in range(N):
            kx.append(-((4 * np.pi) / (3 * np.sqrt(3))) + (
                    ((6 * np.pi) / (3 * np.sqrt(3))) * (j / (N - 1))))
            ky.append((-((2 * np.pi) / 3)) * (j / (N - 1)))

        # from K' to Gamma
        for j in range(N):
            kx.append(((2 * np.pi) / (3 * np.sqrt(3))) * ((N - j - 1) / (N - 1)))
            ky.append(-(2 * np.pi) / 3 * ((N - j - 1) / (N - 1)))

        return kx, ky

    def Plot_K_path_s(self, ax, N):
        """
        Plot energy spectrum along path
        :param ax: axis of plot window
        :param N: number of evaluation points
        :return: figure
        """

        # Get path of k values
        kx, ky = self.path_theta_0(N)
        ev = []

        # Diagonalize Hamiltonian at these k values
        for i in range(len(kx)):
            ev.append(eigvalsh(self.Hamiltonian.get_Hamiltonian(kx[i], ky[i])))

        ev = np.array(ev)
        ev = ev.reshape((3 * N, len(self.Hamiltonian.lattice.orb)))

        axis = np.arange(0, 3 * N, 1)

        fo = open(os.path.join('/tmp', "Bands_path.TXT"), "w")

        for ik in range(len(kx)):
            for ie in range(len(self.Hamiltonian.lattice.orb)):
                fo.write(str(ik) + "   " + str(ev[ik, ie]))
                fo.write("\n")
        fo.close()

        if self.Hamiltonian.lattice.layer == 1:
            ax.plot(axis, ev[:, 0], color="b")
            ax.plot(axis, ev[:, 1], color="y")
        elif self.Hamiltonian.lattice.layer == 2:
            ax.plot(axis, ev[:, 0], color="b")
            ax.plot(axis, ev[:, 1], color="y")
            ax.plot(axis, ev[:, 2], color="g")
            ax.plot(axis, ev[:, 3], color="r")

        else:
            ax.plot(axis, ev[:, 0])
            ax.plot(axis, ev[:, 1])
            ax.plot(axis, ev[:, 2])
            ax.plot(axis, ev[:, 3])
            ax.plot(axis, ev[:, 4])
            ax.plot(axis, ev[:, 5])



            #ax.axhline(y=0, color='black')

        ax.set_title('Plot along path in k-space', fontsize=10)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xticklabels(["$\Gamma$", "$K$", "$K'$", "$\Gamma$"])
        ax.set_ylabel("$\epsilon_{k}$")
        ax.set_yticks([-5, 0, 5])

    def Berry_path_plot(self, angle, N, typ, ax = None):
        """
        Plot Berry curvature as colormap of bands
        :param angle: field shift angle for K-path
        :param N: number of evaluation points
        :param typ: which bands to color, choose from: 'middle','all' & 'occ/unocc'
        :param ax: axis object if applicable, otherwise function will create one
        :return: Plot of Berry curvature
        """

        if self.Hamiltonian.angle == 0:

            kx, ky = path_theta_0(N)

        elif self.Hamiltonian.angle == np.pi/6:
            kx, ky = path_theta_pi_6(N)

        else:

            kx, ky = path(N)
        ev = []
        bc = []
        bc_o = np.zeros(len(kx))
        bc_u = np.zeros(len(kx))

        for i in range(len(kx)):
            ev.append(eigvalsh(H.get_Hamiltonian(kx[i], ky[i])))
            if typ == 'middle':
                bc.append(inv.berry_curvature_middle_states(np.array([kx[i], ky[i]]), 0.05))
            elif typ == 'all':
                bc.append(inv.berry_curvature_per_band(np.array([kx[i], ky[i]]),0.05))
            else:
                bc_o[i] = inv.berry_curvature(np.array([kx[i], ky[i]]),0.05)
                bc_u[i] = inv.berry_curvature_unoccupied(np.array([kx[i], ky[i]]), 0.05)

        bc = np.array(bc)

        ev = np.array(ev)
        ev = ev.reshape((3 * N, len(H.lattice.orb)))

        axis = np.arange(0, 3 * N, 1)



        CMAP = plt.get_cmap('coolwarm')
        s = len(axis)

        def myCMAP(x):
            return CMAP((s * x + 1) / 2)

        if not ax:
            fig = plt.figure(figsize=(16, 8))
            ax0 = fig.add_subplot(1, 1, 1)
        else:
            ax0 = ax

        if typ == 'middle':
            colors = [myCMAP(v) for v in bc]
            # if i == 1 or i == 2:
            #    colors = 'lightgrey'
            ax0.scatter(axis, ev[:, 1], c=colors, s=4.0, rasterized=True)
            ax0.scatter(axis, ev[:, 2], c=colors, s=4.0, rasterized=True)

        elif typ == 'all':
            for i in range(len(bc[0, :])):
                colors = [myCMAP(v) for v in bc[:, i]]
                # if i == 1 or i == 2:
                #    colors = 'lightgrey'
                ax0.scatter(axis, ev[:, i], c=colors, s=4.0, rasterized=True)

        else:

            colors_o = [myCMAP(v) for v in bc_o]
            colors_u = [myCMAP(v) for v in bc_u]
            ax0.scatter(axis, ev[:,1], c=colors_o, s=4.0, rasterized=True)
            ax0.scatter(axis, ev[:,2], c=colors_u, s=4.0, rasterized=True)




        ax0.set_title('Plot along path in k-space', fontsize=10)
        ax0.set_xticks([0, N, 2 * N, 3 * N])
        ax0.set_xticklabels(["$\Gamma$", "$K$", "$K'$", "$\Gamma$"])

        ax0.set_ylabel("$\epsilon_{k}$")
        ax0.set_yticks([-5, 0, 5])
        txt = 'B = {}'.format(B)
        ax0.text(1.5 * N, 5.5, txt, ha="center", va="center", bbox=dict(boxstyle="square", ec='black', fc='white'))

        if not ax:
            ax0.show()







    def summary(self, N, k_x, k_y, x, y, dx, dy):
        """
        Generates summary plot with multiple figures
        :param N: Number of k points
        :param k_x: kx momentum for zoomed plot
        :param k_y: ky momentum for zoomed plot
        :param x: coordinate for plot intersection x
        :param y: coordinate for plot intersection y
        :param dx: coordinate for zoomed plot intersection x
        :param dy: coordinate for zoomed plot intersection y
        :return: summary plot
        """
        # Diagonalize full Hamiltonian
        self.BZ = self.Hamiltonian.energy(N)


        fig = plt.figure(figsize=(16, 8), num='Summary Plot')


        ax1 = fig.add_subplot(241)
        self.Plot_Fermi_surface_s(ax1, x, y)

        ax2 = fig.add_subplot(242)
        self.Plot_K_path_s(ax2, 50)

        ax3 = fig.add_subplot(243)
        self.Plot_along_kx_s(ax3, y, self.BZ)

        ax4 = fig.add_subplot(244)
        self.Plot_along_ky_s(ax4, x, self.BZ)

        # Diagonalize zoomed Hamiltonian
        self.K = self.Hamiltonian.energy(N, k_x - 0.4, k_x + 0.4, k_y - 0.4, k_y + 0.4)

        fo = open(os.path.join('/tmp', "Bands_zoomed_y.TXT"), "w")



        for ik in range(len(self.Hamiltonian.axis1[0])):
            for ie in range(len(self.Hamiltonian.lattice.orb)):
                fo.write(str(ik) + "   " + str(self.K[ik, dy, ie]))
                fo.write("\n")
        fo.close()

        ax5 = fig.add_subplot(245)
        self.Plot_Fermi_surface_zoomed_s(ax5, dx, dy)

        ax6 = fig.add_subplot(246)
        self.Plot_along_ky_s(ax6, dx, self.K)

        ax7 = fig.add_subplot(247)
        self.Plot_along_kx_s(ax7, dy, self.K)

        ax8 = fig.add_subplot(248)
        self.Plot_Fermi_surface_zoomed_one_band_s(ax8)

        plt.show()

    def cmpa_lower_edge(self, wf, kx):
        op = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        if self.Hamiltonian.lattice.layer == 1:
            if self.Hamiltonian.lattice.stack == 'ZZ':
                op[0, 0] = 1
                op[1, 1] = 1
            else:
                op[0, 0] = 1
                op[1, 1] = 1
                op[2, 2] = 1
                op[3, 3] = 1
        else:
            if (self.Hamiltonian.lattice.stack == 'ZZ AA' or self.Hamiltonian.lattice.stack == 'ZZ AB'):
                op[0, 0] = 1
                op[1, 1] = 1
                op[2, 2] = 1
                op[3, 3] = 1
            else:
                op[0, 0] = 1
                op[1, 1] = 1
                op[2, 2] = 1
                op[3, 3] = 1
                op[4, 4] = 1
                op[5, 5] = 1
                op[6, 6] = 1
                op[7, 7] = 1

        res = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):
            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res[i, j] = np.matmul(left, np.matmul(op, right)) / np.matmul(left, right)
        return res

    def y_position_operator_layer_polarized(self, wf, kx):
        op_up = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        op_low = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))

        for i in range(len(self.Hamiltonian.lattice.orb)):
            if i%2 == 0:
                op_low[i,i] = self.Hamiltonian.lattice.orb[i, 1]
            else:
                op_up[i,i] = self.Hamiltonian.lattice.orb[i, 1]

        #op = np.diag(self.Hamiltonian.lattice.orb[:, 1])

        res_up = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_up[i, j] = np.matmul(left, np.matmul(op_up, right)) / np.matmul(left, right)

        res_low = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_low[i, j] = np.matmul(left, np.matmul(op_low, right)) / np.matmul(left, right)

        return np.real(res_up), np.real(res_low)

    def y_position_operator_edge_polarized(self, wf, kx):
        op_eu = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        op_el = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))

        for i in range(len(self.Hamiltonian.lattice.orb)):
            if self.Hamiltonian.lattice.orb[i, 1] > 0:
                op_eu[i, i] = self.Hamiltonian.lattice.orb[i, 1]
            else:
                op_el[i, i] = self.Hamiltonian.lattice.orb[i, 1]

        #op = np.diag(self.Hamiltonian.lattice.orb[:, 1])

        res_up = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_up[i, j] = np.matmul(left, np.matmul(op_eu, right)) / np.matmul(left, right)

        res_low = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_low[i, j] = np.matmul(left, np.matmul(op_el, right)) / np.matmul(left, right)

        return np.real(res_up), np.real(res_low)


    def y_position_operator_fully_polarized(self, wf, kx):
        op_u_d = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        op_u_u = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        op_d_u = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        op_d_d = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))

        for i in range(len(self.Hamiltonian.lattice.orb)):
            if self.Hamiltonian.lattice.orb[i, 1] > 0:
                if i%2 == 0:
                    op_u_d[i, i] = self.Hamiltonian.lattice.orb[i, 1]
                else:
                    op_u_u[i, i] = self.Hamiltonian.lattice.orb[i, 1]
            else:
                if i%2 == 0:
                    op_d_d[i, i] = self.Hamiltonian.lattice.orb[i, 1]
                else:
                    op_d_u[i, i] = self.Hamiltonian.lattice.orb[i, 1]



        res_u_u = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')
        res_u_d = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')
        res_d_u = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')
        res_d_d = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_u_u[i, j] = np.matmul(left, np.matmul(op_u_u, right)) / np.matmul(left, right)

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_u_d[i, j] = np.matmul(left, np.matmul(op_u_d, right)) / np.matmul(left, right)

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_d_d[i, j] = np.matmul(left, np.matmul(op_d_d, right)) / np.matmul(left, right)

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res_d_u[i, j] = np.matmul(left, np.matmul(op_d_u, right)) / np.matmul(left, right)

        return np.real(res_u_u), np.real(res_u_d), np.real(res_d_u), np.real(res_d_d)


    def y_position_operator(self, wf, kx):
        op = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        op = np.diag(self.Hamiltonian.lattice.orb[:, 1])

        res = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res[i, j] = np.matmul(left, np.matmul(op, right)) / np.matmul(left, right)

        return np.real(res)

    def cmpa_upper_edge(self, wf, kx):
        op = np.zeros((len(self.Hamiltonian.lattice.orb), len(self.Hamiltonian.lattice.orb)))
        if self.Hamiltonian.lattice.layer == 1:
            if self.Hamiltonian.lattice.stack == 'ZZ':
                op[-1, -1] = 1
                op[-2, -2] = 1
            else:
                op[-1, -1] = 1
                op[-2, -2] = 1
                op[-3, -3] = 1
                op[-4, -4] = 1
        else:
            if (self.Hamiltonian.lattice.stack == 'ZZ AA' or self.Hamiltonian.lattice.stack == 'ZZ AB'):
                op[-1, -1] = 1
                op[-2, -2] = 1
                op[-3, -3] = 1
                op[-4, -4] = 1
            else:
                op[-1, -1] = 1
                op[-2, -2] = 1
                op[-3, -3] = 1
                op[-4, -4] = 1
                op[-5, -5] = 1
                op[-6, -6] = 1
                op[-7, -7] = 1
                op[-8, -8] = 1

        res = np.zeros((len(kx), len(self.Hamiltonian.lattice.orb)), dtype='complex')

        for i in range(len(kx)):

            for j in range(len(self.Hamiltonian.lattice.orb)):
                left = np.transpose(np.conjugate(wf[i, :, j]))
                right = wf[i, :, j]
                res[i, j] = np.matmul(left, np.matmul(op, right)) / np.matmul(left, right)
        return res

    def show_rib(self, kmax = 2 * np.pi / (np.sqrt(3))):

        E = self.Hamiltonian.energy(100, 0, kmax)



        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 2 * np.pi / (np.sqrt(3)), np.min(E) - 0.1, edgecolor='none', facecolor='grey',
                                 alpha=0.3, zorder=0)
        ax.add_patch(rect)

        for i in range(len(self.Hamiltonian.lattice.orb)):
            plt.scatter(self.Hamiltonian.axis1, E[:, i], color='k', s=8)
            plt.xticks([0, 2 * np.pi / (3 * np.sqrt(3)), np.pi / (np.sqrt(3)), 4*np.pi / (3*np.sqrt(3)), 2 * np.pi / (np.sqrt(3))], ('$\Gamma$', 'K', 'X', "K'", '$\Gamma$'))

        ax.axvline(4*np.pi / (3*np.sqrt(3)), color='black')
        ax.axvline(2 * np.pi / (3 * np.sqrt(3)), color='black')

        plt.ylabel('E/t')
        fig.canvas.set_window_title('Bandstructure')
        plt.show()

    def lower_edge(self):

        kx = np.linspace(0, 2 * np.pi / (np.sqrt(3)), 100)

        EV = []
        wfunc = []
        k = []


        for x in kx:
            evals, ef = self.Hamiltonian.get_wf(x)
            EV.append(evals)
            wfunc.append(ef)


        wf = np.array(wfunc)
        E = np.array(EV)

        for i in range(len(self.Hamiltonian.lattice.orb)):
            k.append(kx)

        k = np.array(k).T.tolist()

        #up, low = self.y_position_operator_edge_polarized(wf, kx)

        uu,ud,du,dd = self.y_position_operator_fully_polarized(wf, kx)
        #up = np.abs(self.cmpa_upper_edge(wf, kx))
        #low = np.abs(self.cmpa_lower_edge(wf, kx))



        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 2 * np.pi / (np.sqrt(3)), np.min(E) - 0.1, edgecolor='none', facecolor='grey',
                                 alpha=0.3, zorder=0)
        ax.add_patch(rect)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0., 'black'), (1.0, 'blue')])

        img = plt.scatter(k, EV, s=8, c=du, cmap='Reds_r', vmin=self.Hamiltonian.lattice.orb[0, 1], vmax=0)
        plt.xticks([0, np.pi / (np.sqrt(3)), 2 * np.pi / (np.sqrt(3))], ('$\Gamma$', 'X', '$\Gamma$'))

        cbar = plt.colorbar(format='%05.2f')
        cbar.set_label('y coordinate', rotation=270, labelpad=15)
        cbar.set_norm(Normalize(vmin=self.Hamiltonian.lattice.orb[0, 1], vmax=0))
        

        plt.ylabel('E/t')
        fig.canvas.set_window_title('Edge states lower edge')
        plt.show()


    def upper_edge(self):
        kx = np.linspace(0, 2 * np.pi / (np.sqrt(3)), 100)

        EV = []
        wfunc = []
        k = []

        for x in kx:
            evals, ef = self.Hamiltonian.get_wf(x)
            EV.append(evals)
            wfunc.append(ef)

        wf = np.array(wfunc)
        E = np.array(EV)

        for i in range(len(self.Hamiltonian.lattice.orb)):
            k.append(kx)

        k = np.array(k).T.tolist()

        uu, ud, du, dd = self.y_position_operator_fully_polarized(wf, kx)
        #up,low = self.y_position_operator_edge_polarized(wf, kx)
        #low = np.abs(self.cmpa_lower_edge(wf, kx))

        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 2 * np.pi / (np.sqrt(3)), np.min(E) - 0.1, edgecolor='none', facecolor='grey',
                                 alpha=0.3, zorder=0)
        ax.add_patch(rect)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0., 'black'), (1.0, 'red')])

        #for i in range(len(self.Hamiltonian.lattice.orb)):

        img = plt.scatter(k, EV, s=8, c=uu, cmap='Blues', vmin=0, vmax=self.Hamiltonian.lattice.orb[-1, 1])
        plt.xticks([0, np.pi / (np.sqrt(3)), 2 * np.pi / (np.sqrt(3))], ('$\Gamma$', 'X', '$\Gamma$'))

        cbar = plt.colorbar(format='%05.2f')
        cbar.set_label('y coordinate', rotation=270, labelpad=15)
        cbar.set_norm(Normalize(vmin=0, vmax=self.Hamiltonian.lattice.orb[-1, 1]))
        


        plt.ylabel('E/t')
        fig.canvas.set_window_title('Edge states upper edge')
        plt.show()


    def calc_edge(self):
        kx = np.linspace(0, 2 * np.pi / (np.sqrt(3)), 100)

        EV = []
        wfunc = []
        k = []

        for x in kx:
            evals, ef = self.Hamiltonian.get_wf(x)
            EV.append(evals)
            wfunc.append(ef)

        for i in range(len(self.Hamiltonian.lattice.orb)):
            k.append(kx)

        k = np.array(k).T.tolist()
        wf = np.array(wfunc)
        E = np.array(EV)

        ccmap = self.y_position_operator(wf, kx)
        fo = open(os.path.join('/tmp', "Bands.TXT"), "w")

        for ik in range(len(kx)):
            for ie in range(len(self.Hamiltonian.lattice.orb)):
                fo.write(str(kx[ik]) + "   " + str(E[ik, ie]) + "   " + str(ccmap[ik, ie]))
                fo.write("\n")
        fo.close()


        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 2 * np.pi / (np.sqrt(3)), np.min(E) - 0.1, edgecolor='none', facecolor='grey',
                                 alpha=0.3, zorder=0)
        ax.add_patch(rect)


        img = plt.scatter(k, EV, s=10, c=ccmap, cmap='rainbow', vmin=self.Hamiltonian.lattice.orb[0, 1], vmax=self.Hamiltonian.lattice.orb[-1, 1])
        #edgecolors='black', linewidth=0.5
        plt.xticks([0, np.pi / (np.sqrt(3)), 2 * np.pi / (np.sqrt(3))], ('$\Gamma$', 'X', '$\Gamma$'))

        cbar = plt.colorbar(format='%05.2f')
        cbar.set_label('y coordinate', rotation=270, labelpad=15)
        cbar.set_norm(Normalize(vmin=self.Hamiltonian.lattice.orb[0, 1], vmax=self.Hamiltonian.lattice.orb[-1, 1]))
        

        



        plt.ylabel('E/t')
        fig.canvas.set_window_title('Edge states')
        plt.show()


    def calc_layer_pol(self):
        kx = np.linspace(0, 2 * np.pi / (np.sqrt(3)), 100)

        EV = []
        wfunc = []
        k = []

        for x in kx:
            evals, ef = self.Hamiltonian.get_wf(x)
            EV.append(evals)
            wfunc.append(ef)

        for i in range(len(self.Hamiltonian.lattice.orb)):
            k.append(kx)

        k = np.array(k).T.tolist()
        wf = np.array(wfunc)
        E = np.array(EV)

        ccmap_u, ccmap_l = self.y_position_operator_layer_polarized(wf, kx)


        fig = plt.figure(figsize=(12, 6), num='Layer Polarization')

        ax1 = fig.add_subplot(121)

        rect = patches.Rectangle((0, 0), 2 * np.pi / (np.sqrt(3)), np.min(E) - 0.1, edgecolor='none', facecolor='grey',
                                 alpha=0.3, zorder=0)
        ax1.add_patch(rect)


        img = ax1.scatter(k, EV, s=10, c=ccmap_u, cmap='rainbow', vmin=self.Hamiltonian.lattice.orb[0, 1], vmax=self.Hamiltonian.lattice.orb[-1, 1])
        
        ax1.set_xticks([0, np.pi / (np.sqrt(3)), 2 * np.pi / (np.sqrt(3))])
        ax1.set_xticklabels(['$\Gamma$', 'X', '$\Gamma$'])
        ax1.set_title('Upper Layer')
        



        ax1.set_ylabel('E/t')
        

        ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

        rect = patches.Rectangle((0, 0), 2 * np.pi / (np.sqrt(3)), np.min(E) - 0.1, edgecolor='none', facecolor='grey',
                                 alpha=0.3, zorder=0)
        ax2.add_patch(rect)

        img = ax2.scatter(k, EV, s=10, c=ccmap_l, cmap='rainbow', vmin=self.Hamiltonian.lattice.orb[0, 1],
                          vmax=self.Hamiltonian.lattice.orb[-1, 1])
        
        ax2.set_xticks([0, np.pi / (np.sqrt(3)), 2 * np.pi / (np.sqrt(3))])
        ax2.set_xticklabels(['$\Gamma$', 'X', '$\Gamma$'])

        
        ax2.set_ylabel('E/t')
        ax2.set_title('Lower Layer')
        plt.show()
