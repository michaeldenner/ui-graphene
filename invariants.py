# Import common packages
import numpy as np
from scipy import integrate
from numpy.linalg import eigvalsh, eigh

"""
def evaluation_points(k, r, s, phi=0, t6=0, t7=0):
    
    Calculate evaluation points for Berry phase
    :param k: Number of evaluation points
    :param r: Radius around K-point
    :param phi: Magnetic field strength
    :param t6: Inter-layer hopping t6
    :param t7: Inter-layer hopping t7
    :param s: Sign to calculate eval points around upper or lower K point
    :return: Evaluation points
    

    eval_points = np.zeros((k, 2))

    if phi == 0:
        shift_x = 0
    else:
        shift_x = -phi / 2 + s * np.sqrt(((t6 * t7) / (4.5 * 4.5)) + ((phi * phi) / 4))

    for n in range(0, k):
        theta = ((2 * np.pi) / k) * n
        ix = (150 / 0.8) * (r * np.cos(theta) + 0.4)
        iy = (150 / 0.8) * (r * np.sin(theta) + 0.4 + shift_x)
        eval_points[n, 0] = np.round(ix, 0)
        eval_points[n, 1] = np.round(iy, 0)

    return eval_points
"""

class topology:
    """
    This class computes topological invariants
    """
    def __init__(self, Hamiltonian):
        """
        Initialize object
        :param Hamiltonian: Hamiltonian of the system to be studied
        """
        self.Hamiltonian = Hamiltonian

    def edge_states(self, kx):
        """
        Function returning the edge states of a k point
        :param kx: kx momentum
        :return: array of edge states
        """
        es, efs = self.Hamiltonian.get_wf(kx)

        efs = np.conjugate(efs.transpose())





    def occupied_states(self, kx, ky):
        """
        Function returning the occupied states of a k point
        :param kx: kx momentum
        :param ky: ky momentum
        :return: array of occupied states
        """
        es, efs = self.Hamiltonian.get_wf(kx, ky)  # Get wavefunction and energies

        efs = np.conjugate(efs.transpose())
        occef = []
        for (ie, iw) in zip(es, efs): # Go through all energies and wavefunctions

            if ie < 0:  # if below fermi
                occef.append(iw)
        return np.array(occef) # return array of occupied states


    def unoccupied_states(self, kx, ky):
        """
        Function returning the unoccupied states of a k point
        :param kx: kx momentum
        :param ky: ky momentum
        :return: array of unoccupied states
        """
        es, efs = self.Hamiltonian.get_wf(kx, ky)  # Get wavefunction and energies

        efs = np.conjugate(efs.transpose())
        uoccef = []
        for (ie, iw) in zip(es, efs): # Go through all energies and wavefunctions

            if ie > 0:  # if above fermi
                uoccef.append(iw)
        return np.array(uoccef) # return array of unoccupied states

    def middle_states(self, kx, ky):
        """
        Function returning the middle states of a k point
        :param kx: kx momentum
        :param ky: ky momentum
        :return: array of middle states
        """
        es, efs = self.Hamiltonian.get_wf(kx, ky)  # Get wavefunction and energies

        efs = np.conjugate(efs.transpose())
        mef = []


        for (ie, iw) in zip(es, efs): # Go through all energies and wavefunctions

            if (ie > np.min(es) and ie < np.max(es)):  # if around fermi
                mef.append(iw)
        return np.array(mef) # return array of unoccupied states


    def mij(self, wf1, wf2):
        """
        Calculates the matrix product of two sets of input wavefunctions
        :param wf1: wavefunction 1
        :param wf2: wavefunction 2
        :return: matrix product of input
        """
        out = np.matrix(np.conjugate(wf1)) @ (np.matrix(wf2).T)
        return out


    def berry_curvature(self, k, dk=0.01):
        """
        Calculates the Berry curvature
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """


        kx = k[0]
        ky = k[1]



        # get wavefunctions

        wf1 = self.occupied_states(kx - dk, ky - dk)
        wf2 = self.occupied_states(kx + dk, ky - dk)
        wf3 = self.occupied_states(kx + dk, ky + dk)
        wf4 = self.occupied_states(kx - dk, ky + dk)

        # get the mij
        m = self.mij(wf1, wf2) @ self.mij(wf2, wf3) @ self.mij(wf3, wf4) @ self.mij(wf4, wf1)
        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi

    def berry_curvature_line_x(self, kx, dk=0.01):
        """
        Calculates the Berry curvature along a line
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """

        kvals = np.arange(0, 2*np.pi, dk)

        for i in range(len(kvals)):
            # get wavefunctions

            if i == 0:

                wf1 = self.occupied_states(kx, kvals[i])
                wf2 = self.occupied_states(kx, kvals[i+1])

                m = self.mij(wf1, wf2)

            elif i == len(kvals)-1:

                wf1 = self.occupied_states(kx, kvals[i])
                wf2 = self.occupied_states(kx, kvals[0])

                m = m @ self.mij(wf1, wf2)

            else:

                wf1 = self.occupied_states(kx, kvals[i])
                wf2 = self.occupied_states(kx, kvals[i+1])

                m = m @ self.mij(wf1, wf2)


        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi

    def berry_curvature_line_edge(self, dk=0.01):
        """
        Calculates the Berry curvature along a line
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """

        kvals = np.arange(0, 2 * np.pi / (np.sqrt(3)), dk)

        for i in range(len(kvals)):
            # get wavefunctions

            if i == 0:

                wf1 = self.edge_states(kvals[i])
                wf2 = self.edge_states(kvals[i+1])

                m = self.mij(wf1, wf2)

            elif i == len(kvals)-1:

                wf1 = self.edge_states(kvals[i])
                wf2 = self.edge_states(kvals[0])

                m = m @ self.mij(wf1, wf2)

            else:

                wf1 = self.edge_states(kvals[i])
                wf2 = self.edge_states(kvals[i+1])

                m = m @ self.mij(wf1, wf2)


        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi

    def berry_curvature_line_B(self, kx, ky, dB=0.01):
        """
        Calculates the Berry curvature along a line
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """

        Bvals = np.arange(0, 2*np.pi, dB)

        for i in range(len(Bvals)):
            # get wavefunctions

            if i == 0:
                self.Hamiltonian.add_magnetic_field(Bvals[i], 'In-plane y')
                wf1 = self.occupied_states(kx, ky)
                self.Hamiltonian.add_magnetic_field(Bvals[i+1], 'In-plane y')
                wf2 = self.occupied_states(kx, ky)

                m = self.mij(wf1, wf2)

            elif i == len(Bvals)-1:

                self.Hamiltonian.add_magnetic_field(Bvals[i], 'In-plane y')
                wf1 = self.occupied_states(kx, ky)
                self.Hamiltonian.add_magnetic_field(Bvals[0], 'In-plane y')
                wf2 = self.occupied_states(kx, ky)

                m = m @ self.mij(wf1, wf2)

            else:

                self.Hamiltonian.add_magnetic_field(Bvals[i], 'In-plane y')
                wf1 = self.occupied_states(kx, ky)
                self.Hamiltonian.add_magnetic_field(Bvals[i+1], 'In-plane y')
                wf2 = self.occupied_states(kx, ky)

                m = m @ self.mij(wf1, wf2)


        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi

    def berry_curvature_line_y(self, ky, dk=0.01):
        """
        Calculates the Berry curvature along a line
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """

        kvals = np.arange(0, 2*np.pi, dk)

        for i in range(len(kvals)):
            # get wavefunctions

            if i == 0:

                wf1 = self.occupied_states(kvals[i], ky)
                wf2 = self.occupied_states(kvals[i + 1], ky)

                m = self.mij(wf1, wf2)

            elif i == len(kvals)-1:

                wf1 = self.occupied_states(kvals[i], ky)
                wf2 = self.occupied_states(kvals[0], ky)

                m = m @ self.mij(wf1, wf2)

            else:

                wf1 = self.occupied_states(kvals[i], ky)
                wf2 = self.occupied_states(kvals[i + 1], ky)

                m = m @ self.mij(wf1, wf2)

        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real)  # return phase
        return phi

    def berry_curvature_unoccupied(self, k, dk=0.01):
        """
        Calculates the Berry curvature of unoccupied states
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """

        kx = k[0]
        ky = k[1]


        # get wavefunctions

        wf1 = self.unoccupied_states(kx - dk, ky - dk)
        wf2 = self.unoccupied_states(kx + dk, ky - dk)
        wf3 = self.unoccupied_states(kx + dk, ky + dk)
        wf4 = self.unoccupied_states(kx - dk, ky + dk)

        # get the mij
        m = self.mij(wf1, wf2) @ self.mij(wf2, wf3) @ self.mij(wf3, wf4) @ self.mij(wf4, wf1)
        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi

    def berry_curvature_middle_states(self, k, dk=0.01):
        """
        Calculates the Berry curvature of middle states
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """


        kx = k[0]
        ky = k[1]



        # get wavefunctions

        wf1 = self.middle_states(kx - dk, ky - dk)
        wf2 = self.middle_states(kx + dk, ky - dk)
        wf3 = self.middle_states(kx + dk, ky + dk)
        wf4 = self.middle_states(kx - dk, ky + dk)

        # get the mij
        m = self.mij(wf1, wf2) @ self.mij(wf2, wf3) @ self.mij(wf3, wf4) @ self.mij(wf4, wf1)
        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi

    def berry_curvature_per_band(self, k, dk=0.01):
        """
        Calculates the Berry curvature per band
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Berry phase
        """


        kx = k[0]
        ky = k[1]



        # get wavefunctions



        e1, ef1 = self.Hamiltonian.get_wf(kx - dk, ky - dk)
        e2, ef2 = self.Hamiltonian.get_wf(kx + dk, ky - dk)
        e3, ef3 = self.Hamiltonian.get_wf(kx + dk, ky + dk)
        e4, ef4 = self.Hamiltonian.get_wf(kx - dk, ky + dk)

        ef1 = np.conjugate(ef1.transpose())
        ef2 = np.conjugate(ef2.transpose())
        ef3 = np.conjugate(ef3.transpose())
        ef4 = np.conjugate(ef4.transpose())

        phi = []

        for i in range(len(self.Hamiltonian.lattice.orb)):
            wf1 = ef1[i]
            wf2 = ef2[i]
            wf3 = ef3[i]
            wf4 = ef4[i]

            # get the mij
            m = self.mij(wf1, wf2) @ self.mij(wf2, wf3) @ self.mij(wf3, wf4) @ self.mij(wf4, wf1)
            d = np.linalg.det(m)  # calculate determinant
            phi.append(np.arctan2(d.imag, d.real)) # return phase

        return phi

    """
    def berry_curvature_path(H, n, r, loc, phi=0, t6=0, t7=0):
         Calculates the Berry curvature along a closed circle in k-space
    
        eval_points = evaluation_points(n, r, loc, phi, t6, t7)
    
    
        # get wavefunctions
        for i in range(n):
            if i == n - 1:
                wf1 = occupied_states(H, np.array([eval_points[i, 0], eval_points[i, 1]]))
                wf2 = occupied_states(H, np.array([eval_points[0, 0], eval_points[0, 1]]))
            else:
                wf1 = occupied_states(H, np.array([eval_points[i, 0], eval_points[i, 1]]))
                wf2 = occupied_states(H, np.array([eval_points[i + 1, 0], eval_points[i + 1, 1]]))
    
            if i == 0:
                m = mij(wf1, wf2)
            else:
                m = m @ mij(wf1, wf2)
    
        d = np.linalg.det(m)  # calculate determinant
        phi = np.arctan2(d.imag, d.real) # return phase
        return phi, eval_points
    """

    def chern_loc(self, k, dk=0.01):

        """
        Calculates the local Chern number
        :param k: momentum k
        :param dk: optional spacing of evaluation momenta
        :return: Local Chern number
        """


        c = self.berry_curvature(k, dk)

        return c / (2. * np.pi * 4 * dk * dk)










    def chern_int(self, dk):
        """
        Calculates the Chern number
        :param dk: spacing of evaluation momenta
        :return: Chern number
        """
        def f(x, y):  # function to integrate

            return self.berry_curvature(np.array([x, y]), dk)/(dk)**2

        c = integrate.dblquad(f, -np.pi, np.pi, lambda x: -np.pi, lambda x: np.pi, epsabs=0.01, epsrel=0.01) # Integrate over BZ

        return int(c[0] / (2. * np.pi)**2)

    def chern_valley(self, valley, dk):

        """
        Calculates the chern number of a valley
        :param valley: valley used for calculation
        :param dk: spacing of evaluation momenta
        :return: Valley Chern number
        """

        delta = 0.5

        def f(x, y):  # function to integrate

            return self.berry_curvature(np.array([x, y]), dk)/(4*dk*dk)

        if valley == 'K':
            # Position of K point
            K = np.array([-(4*np.pi)/(3*np.sqrt(3)), 0])

            # Integrate over small region
            c1 = integrate.dblquad(f, K[1]-delta, K[1]+delta, lambda x: K[0]-delta, lambda x: K[0]+delta, epsabs=0.01, epsrel=0.01)

            # Position of K point
            K = np.array([(4 * np.pi) / (6 * np.sqrt(3)), (4 * np.pi) / (3 * 2)])

            # Integrate over small region
            c2 = integrate.dblquad(f, K[1] - delta, K[1] + delta, lambda x: K[0] - delta, lambda x: K[0] + delta,
                                   epsabs=0.01, epsrel=0.01)

            # Position of K point
            K =np.array([(4 * np.pi) / (6 * np.sqrt(3)), -(4 * np.pi) / (3 * 2)])

            # Integrate over small region
            c3 = integrate.dblquad(f, K[1] - delta, K[1] + delta, lambda x: K[0] - delta, lambda x: K[0] + delta,
                                   epsabs=0.01, epsrel=0.01)

        elif valley == "K'":
            # Position of K' point
            K = np.array([(4*np.pi)/(3*np.sqrt(3)), 0])

            # Integrate over small region
            c1 = integrate.dblquad(f, K[1]-delta, K[1]+delta, lambda x: K[0]-delta, lambda x: K[0]+delta, epsabs=0.01, epsrel=0.01)

            # Position of K' point
            K = np.array([-(4 * np.pi) / (6 * np.sqrt(3)), (4 * np.pi) / (3 * 2)])

            # Integrate over small region
            c2 = integrate.dblquad(f, K[1] - delta, K[1] + delta, lambda x: K[0] - delta, lambda x: K[0] + delta,
                                   epsabs=0.01, epsrel=0.01)

            # Position of K' point
            K =np.array([-(4 * np.pi) / (6 * np.sqrt(3)), -(4 * np.pi) / (3 * 2)])

            # Integrate over small region
            c3 = integrate.dblquad(f, K[1] - delta, K[1] + delta, lambda x: K[0] - delta, lambda x: K[0] + delta,
                                   epsabs=0.01, epsrel=0.01)

        return (c1[0]+c2[0]+c3[0]) / (4*2*np.pi*delta*delta)
