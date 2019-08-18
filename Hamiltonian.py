import numpy as np
from numpy.linalg import eigvalsh, eigh
from scipy.sparse.linalg import eigsh
import sys
import os




class Hamiltonian:
    """
    This class generates the Hamiltonian corresponding to the lattice geometry defined in Lattice.py
    """
    def __init__(self, lattice, t, tprime=0, tbrick=0, haldane = 0):
        """
        Initialize object
        :param lattice: lattice object
        :param t: intralayer hopping strength
        :param tprime: optional interlayer hopping strength
        :param haldane: optional haldane hopping
        """
        self.lattice = lattice
        self.intra_layer = t
        self.inter_layer = tprime
        self.haldane = haldane
        self.intra_layer1 = tbrick

    def add_interface(self, i= False):
        self.interface = i



    def add_magnetic_field(self, B, btyp, angle1 = 0, angle2 = 0, B2 = 0):
        """
        Function adding a magnetic field to the lattice
        :param B: magnetic field strength
        :param btyp: magnetic field type
        :param angle1: optional orientation of the magnetic field
        :param angle2: optional orientation of the magnetic field in a trilayer geometry
        :param B2: optional magnetic field strength in a trilayer geometry
        """
        self.B = B
        self.btyp = btyp
        self.angle = angle1
        self.angle2 = angle2
        self.B2 = B2

    def add_nearest_neighbour_hopping(self, nnn):

        """
        Function adding next nearest neighbour hopping
        :param nnn: nnn hopping strength
        """
        self.nnn = nnn

    def add_lattice_imbalance(self, V):

        """
        Function adding a lattice imbalance
        :param V: strength of on-site potentials
        """

        self.V = V

    def add_sublattice_imbalance(self, V):

        """
        Function adding a lattice imbalance
        :param V: strength of on-site potentials
        """

        self.Vs = V

    def add_disorder(self, V):
        """
        Function adding a disorder potential
        :param VD: strength of on-site potential
        """
        self.VD = V

    def add_antiHaldane(self):
        """
        Function adding a disorder potential
        :param VD: strength of on-site potential
        """
        self.anti = -1

    def build_Hamiltonian(self):

        """
        Function building the Hamiltonian for a certain lattice
        """

        def twisted_hop(self, orb1, orb2, add_onsite = False):
            rr = orb1 - orb2
            dx = orb1[0] - orb2[0]
            dy = orb1[1] - orb2[1]
            dz = orb1[2] - orb2[2]
            rr = rr.dot(rr)
            norm = np.sqrt(rr)
            if norm > 25.0:
                return 0.0
            if norm < 0.001:

                if add_onsite:
                    return np.sign(orb1[-1]) * self.V
                else:
                    return 0.0
            else:
                t = -(dx * dx + dy * dy) / rr * np.exp(-12.0 * (norm - 1.0)) * np.exp(-10.0 * dz * dz)
                t += -self.inter_layer * (dz*dz)/rr*np.exp(-8.0*(norm-3.0))

                if hasattr(self, 'B'):
                    t *= peierls(self, orb1, orb2)
            return t

        def vec_pot(self, v):

            """
            Function generating the vector potential for a specific magnetic field
            :param self: object
            :param v: position vector
            :return: Vector potential at position v
            """

            if self.btyp == 'In-plane x':

                # Add in-plane field in x direction

                A = np.array([0, -self.B * v[2]])


            elif self.btyp == 'In-plane y':

                # Add in-plane field in x direction

                A = np.array([self.B * v[2], 0])


            elif self.btyp == 'In-plane angle':

                # Add in-plane field with angle direction

                A = np.array([self.B * np.cos(self.angle) * v[2], -self.B * np.sin(self.angle) * v[2]])

            elif self.btyp == 'In-plane angle Jose':

                # Add in-plane field with angle direction

                A = np.array([self.B * np.sin(self.angle*np.pi) * v[2], -self.B * np.cos(self.angle*np.pi) * v[2]])

            elif self.btyp == 'Artificial':

                # Add artificial field

                A = np.array([self.B * v[2] * v[1], self.B * v[2] * v[0]])


            elif self.btyp == 'Perpendicular':

                # Add perpendicular field in z direction

                A = np.array([self.B * v[1], 0])


            elif self.btyp == 'Artificial Trilayer':

                # Add artificial trilayer field with different field between the two layers

                if v[-1] == 2*np.sqrt(2):
                    A = np.array([self.B2 * np.cos(self.angle2) * v[2]/2, -self.B2 * np.sin(self.angle2) * v[2]/2])
                else:
                    A = np.array([self.B * np.cos(self.angle) * v[2], -self.B * np.sin(self.angle) * v[2]])

            # If none specified, return zero

            else:
                A = np.array([0, 0])

            return A


        def peierls(self, vec1, vec2):

            """
            Function for performing the Peierl's substitution
            :param self: object
            :param vec1: position of orbital 1
            :param vec2: position of orbital 2
            :return: exponential containing the magnetic phase factor
            """

            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            # the midpoint between two sites

            mid = 0.5 * (vec1 + vec2)

            # vector potential
            A = vec_pot(self, mid)


            # integral of (A * dl) from position 1 to position 2
            phase = A[0] * (vec1[0] - vec2[0]) + A[1] * (vec1[1] - vec2[1])

            if len(A) == 3:
                phase = A[0] * (vec1[0] - vec2[0]) + A[1] * (vec1[1] - vec2[1]) + A[2] * (vec1[2] - vec2[2])

            # the Peierls substitution
            return np.exp(1j * phase)



        def hop(self, orb1, orb2):

            """
            Function adding the nearest neighbour hoppings from the position of orbitals
            :param self: object
            :param orb1: position of orbital 1
            :param orb2: position of orbital 2
            :return: matrix with hoppings between n.n. sites
            """

            # Initialize array

            h = np.zeros((len(orb1), len(orb1)), dtype='complex')

            # Go through all orbitals

            for i in range(len(orb1)):
                for j in range(len(orb1)):

                    if (np.linalg.norm(orb1[i] - orb2[j]) < self.lattice.nn*0.99 and i != j):

                        # If orbitals are nearest neighbours

                        if (np.linalg.norm(orb1[i] - orb2[j]) < np.sqrt(2)*1.1 and np.linalg.norm(orb1[i] - orb2[j]) > np.sqrt(2)*0.9 and self.lattice.layer > 1):

                            # If nearest neighbours are in different layers, add inter layer hopping

                            if hasattr(self, 'B'):

                                # If a magnetic field is specified, add a phase

                                h[i, j] = self.inter_layer * peierls(self, orb1[i], orb2[j])
                            else:
                                h[i, j] = self.inter_layer
                        else:

                            # If nearest neighbours are in identical layers, add intra layer hopping

                            if hasattr(self, 'B'):

                                # If a magnetic field is specified, add a phase
                                if np.linalg.norm(orb1[0] - orb2[0]) == 0:
                                    h[i, j] = self.intra_layer * peierls(self, orb1[i], orb2[j])
                                else:
                                    h[i, j] = self.intra_layer * peierls(self, orb1[i], orb2[j])
                            else:
                                if np.linalg.norm(orb1[0] - orb2[0]) == 0:

                                    h[i, j] = self.intra_layer

                                else:
                                    h[i, j] = self.intra_layer

                    elif hasattr(self, 'nnn'):
                        if (np.linalg.norm(orb1[i] - orb2[j]) < np.sqrt(3)+0.1 and np.linalg.norm(orb1[i] - orb2[j]) > np.sqrt(3)-0.1 and i != j):
                            if (i == 3 and j != 0):
                                if hasattr(self, 'B'):
                                    h[j, i] = self.nnn * peierls(self, orb1[i], orb2[j])
                                else:
                                    h[j, i] = self.nnn
                        elif (np.linalg.norm(orb1[i] - orb2[j]) == np.sqrt(6) and i != j):
                            if hasattr(self, 'B'):
                                h[i, j] = self.nnn * peierls(self, orb1[i], orb2[j])
                            else:
                                h[i, j] = self.nnn


            return h

        def hop_twisted(self, orb1, orb2):
            """
            Function adding the nearest neighbour hoppings from the position of orbitals
            :param self: object
            :param orb1: position of orbital 1
            :param orb2: position of orbital 2
            :return: matrix with hoppings between n.n. sites
            """

            # Check if layer polarization is needed

            if hasattr(self, 'V'):
                add_onsite = True
            else:
                add_onsite = False

            # Go through all orbitals

            h = np.array([[twisted_hop(self, r1i, r2j, add_onsite) for r1i in orb1] for r2j in orb2], dtype=np.complex)

            return h


        def hop_haldane(self, orb1, orb2, anti = 1):

            """
            Function adding haldane hopping
            :param self: object
            :param orb1: position of orbital 1
            :param orb2: position of orbital 2
            :return: matrix with haldane hopping
            """

            # Initialize array

            h = np.zeros((len(orb1), len(orb1)), dtype='complex')

            # Go through all orbitals

            if self.lattice.typ == 'Ribbon':
                for i in range(len(orb1)):
                    for j in range(i, len(orb1)):


                        if np.linalg.norm(orb1[i] - orb2[j]) < self.lattice.nn+0.1 and np.linalg.norm(orb1[i] - orb2[j]) > self.lattice.nn-0.1:

                            # If orbitals are next nearest neighbours

                            if self.lattice.sublattice[i] == 0 and self.lattice.sublattice[j] == 0:

                                # Hopping between A sites have positive phase

                                h[i, j] = 0.1 * np.exp(1j * self.haldane)

                            elif self.lattice.sublattice[i] == 1 and self.lattice.sublattice[j] == 1:

                                # Hopping between B sites have negative phase

                                h[i, j] = 0.1 * np.exp(-anti * 1j * self.haldane)

                            else:
                                h[i, j] = 0


                return h + np.transpose(np.conjugate(h))

            else:
                for i in range(len(orb1)):
                    for j in range(len(orb1)):

                        if np.linalg.norm(orb1[i] - orb2[j]) < self.lattice.nn + 0.1 and np.linalg.norm(
                                orb1[i] - orb2[j]) > self.lattice.nn - 0.1:

                            # If orbitals are next nearest neighbours

                            if self.lattice.sublattice[i] == 0 and self.lattice.sublattice[j] == 0:

                                # Hopping between A sites have positive phase

                                h[i, j] = 0.1 * np.exp(1j * self.haldane)

                            elif self.lattice.sublattice[i] == 1 and self.lattice.sublattice[j] == 1:

                                # Hopping between B sites have negative phase

                                h[i, j] = 0.1 * np.exp(-anti * 1j * self.haldane)

                            else:
                                h[i, j] = 0

                return h

        if self.lattice.typ == 'Sheet':

            # If the lattice is an infinite sheet
            if self.lattice.stack == 'Twisted Bilayer':
                self.H_00 = hop_twisted(self, self.lattice.orb, self.lattice.orb)
                self.H_m0 = hop_twisted(self, self.lattice.orb, self.lattice.orb - self.lattice.lat[0])
                self.H_m1 = hop_twisted(self, self.lattice.orb, self.lattice.orb - self.lattice.lat[1])
                self.H_p0 = np.transpose(np.conjugate(self.H_m0))
                self.H_p1 = np.transpose(np.conjugate(self.H_m1))
                self.H_pp0 = hop_twisted(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[0] + self.lattice.lat[1])
                self.H_pm1 = hop_twisted(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[0] - self.lattice.lat[1])
                self.H_mm0 = np.transpose(np.conjugate(self.H_pp0))
                self.H_mp1 = np.transpose(np.conjugate(self.H_pm1))

            else:
                if hasattr(self, 'V'):

                    # Check if an on-site potential should be added, depending on number of layers with different sign

                    if self.lattice.layer>2:
                        self.H_00 = hop(self, self.lattice.orb, self.lattice.orb) + np.diag(
                            np.resize([-self.V, -self.V, self.V], len(self.lattice.orb)))

                    else:
                        self.H_00 = hop(self, self.lattice.orb, self.lattice.orb) + np.diag(
                            np.resize([self.V, -self.V], len(self.lattice.orb)))
                elif hasattr(self, 'Vs'):
                    self.H_00 = hop(self, self.lattice.orb, self.lattice.orb) + np.diag(
                        np.resize([self.Vs, self.Vs, -self.Vs, -self.Vs], len(self.lattice.orb)))



                else:
                    self.H_00 = hop(self, self.lattice.orb, self.lattice.orb)

                # Build Hamiltonian from hopping matrices

                self.H_m0 = hop(self, self.lattice.orb, self.lattice.orb - self.lattice.lat[0])
                self.H_m1 = hop(self, self.lattice.orb, self.lattice.orb - self.lattice.lat[1])
                self.H_p0 = np.transpose(np.conjugate(self.H_m0))
                self.H_p1 = np.transpose(np.conjugate(self.H_m1))

                if self.lattice.stack == 'Haldane':

                    if hasattr(self, 'anti'):
                        self.H_h0u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[0], self.anti)
                        self.H_h1u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[1], self.anti)
                        self.H_h2u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[2], self.anti)
                        self.H_h0d = np.transpose(np.conjugate(self.H_h0u))
                        self.H_h1d = np.transpose(np.conjugate(self.H_h1u))
                        self.H_h2d = np.transpose(np.conjugate(self.H_h2u))

                    else:

                        self.H_h0u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[0])
                        self.H_h1u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[1])
                        self.H_h2u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat[2])
                        self.H_h0d = np.transpose(np.conjugate(self.H_h0u))
                        self.H_h1d = np.transpose(np.conjugate(self.H_h1u))
                        self.H_h2d = np.transpose(np.conjugate(self.H_h2u))




        elif self.lattice.typ == 'Ribbon':

            # If the lattice is a finite ribbon
            if self.lattice.stack == 'Twisted Bilayer':
                self.H_00 = hop_twisted(self, self.lattice.orb, self.lattice.orb)
                self.H_m0 = hop_twisted(self, self.lattice.orb, self.lattice.orb - self.lattice.lat)
                self.H_p0 = hop_twisted(self, self.lattice.orb, self.lattice.orb + self.lattice.lat)




            else:
                if hasattr(self, 'V'):

                    # Check if an on-site potential should be added, depending on number of layers with different sign

                    if hasattr(self, 'interface'):
                        inter = np.ones((len(self.lattice.orb)))
                        inter[int(len(self.lattice.orb)/2):]=-1
                        self.H_00 = hop(self, self.lattice.orb, self.lattice.orb) + np.diag(
                            np.multiply(np.resize([self.V, -self.V], len(self.lattice.orb)), inter))

                    else:
                        self.H_00 = hop(self, self.lattice.orb, self.lattice.orb) + np.diag(
                            np.resize([self.V, -self.V], len(self.lattice.orb)))




                    if hasattr(self, 'VD'):

                        # Check if disorder should be added

                        # Lower layer, lower edge
                        self.H_00[0, 0] = -self.VD

                        # Upper layer, lower edge
                        #self.H_00[1, 1] = self.VD

                        # Lower layer, upper edge
                        #self.H_00[-2, -2] = -self.VD

                        # Upper layer, upper edge
                        #self.H_00[-1, -1] = self.VD

                    elif hasattr(self, 'Vs'):
                        self.H_00 += np.diag(np.resize([-self.Vs, -self.Vs, self.Vs, self.Vs], len(self.lattice.orb)))

                elif hasattr(self, 'Vs'):
                    self.H_00 = hop(self, self.lattice.orb, self.lattice.orb) + np.diag(
                        np.resize([-self.Vs, -self.Vs, self.Vs, self.Vs], len(self.lattice.orb)))

                else:
                    self.H_00 = hop(self, self.lattice.orb, self.lattice.orb)

                self.H_m0 = hop(self, self.lattice.orb, self.lattice.orb - self.lattice.lat)
                self.H_p0 = hop(self, self.lattice.orb, self.lattice.orb + self.lattice.lat)

                if self.lattice.stack == 'Haldane':
                    if hasattr(self, 'anti'):
                        self.H_h0u = hop_haldane(self, self.lattice.orb, self.lattice.orb, self.anti)
                        self.H_h1u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat, self.anti)
                        self.H_h1d = np.transpose(np.conjugate(self.H_h1u))
                    else:
                        self.H_h0u = hop_haldane(self, self.lattice.orb, self.lattice.orb)
                        self.H_h1u = hop_haldane(self, self.lattice.orb, self.lattice.orb + self.lattice.lat)
                        self.H_h1d = np.transpose(np.conjugate(self.H_h1u))

    def get_Hamiltonian(self, kx, ky=0, sbz = False):

        """
        Function building the Hamiltonian from n.n. hopping matrices
        :param kx: momentum kx
        :param ky: momentum ky
        :return: Hamiltonian at specified momentum
        """

        if self.lattice.typ == 'Sheet':

            # Depending on geometry, k is 2 or 3 dimensional
            if (self.lattice.stack == 'Monolayer' or self.lattice.stack == 'Haldane'):
                k = np.array([kx, ky])
            else:
                k = np.array([kx, ky, 0])

            # Build Hamiltonian from hopping matrices

            if sbz == True:

                # Hamiltonian in natural phases

                H_Matrix = self.H_00 + self.H_m0 * np.exp(-1j * kx * 2 * np.pi) + self.H_m1 * np.exp(
                    -1j * ky * 2 * np.pi) + self.H_p0 * np.exp(1j * kx * 2 * np.pi) + self.H_p1 * np.exp(
                    1j * ky * 2 * np.pi)

            else:
                if self.lattice.stack == 'Twisted Bilayer':
                    sys.exit("Not supported. Use square BZ")

                else:
                    H_Matrix = self.H_00 + self.H_m0 * np.exp(-1j * k.dot(self.lattice.lat[0])) + self.H_m1 * np.exp(
                        -1j * k.dot(self.lattice.lat[1])) + self.H_p0 * np.exp(
                        1j * k.dot(self.lattice.lat[0])) + self.H_p1 * np.exp(1j * k.dot(self.lattice.lat[1]))





            if self.lattice.stack == 'Haldane':
                k = np.array([kx, ky])

                # Build Hamiltonian from hopping matrices

                H_Matrix += self.H_h0u * np.exp(1j * k.dot(self.lattice.lat[0])) + self.H_h0d * np.exp(
                    -1j * k.dot(self.lattice.lat[0])) + self.H_h1u * np.exp(
                    1j * k.dot(self.lattice.lat[1])) + self.H_h1d * np.exp(
                    -1j * k.dot(self.lattice.lat[1])) + self.H_h2u * np.exp(
                    1j * k.dot(self.lattice.lat[2])) + self.H_h2d * np.exp(-1j * k.dot(self.lattice.lat[2]))

            elif self.lattice.stack == 'Twisted Bilayer':
                add = self.H_pp0 * np.exp(1j * (kx+ky) * 2 * np.pi) + self.H_pm1 * np.exp(1j * (kx-ky) * 2 * np.pi)
                H_Matrix += add + np.transpose(np.conjugate(add))




        elif self.lattice.typ == 'Ribbon':
            k = np.array([kx, ky])

            # Build Hamiltonian from hopping matrices

            H_Matrix = self.H_00 + self.H_m0 * np.exp(-1j * kx * self.lattice.nn) + self.H_p0 * np.exp(
                1j * kx * self.lattice.nn)

            if self.lattice.stack == 'Haldane':
                H_Matrix += self.H_h0u + self.H_h1u * np.exp(
                    1j * k.dot(self.lattice.lat)) + self.H_h1d * np.exp(
                    -1j * k.dot(self.lattice.lat))

            if self.lattice.stack == 'Twisted Bilayer':
                H_Matrix = self.H_00 + self.H_m0 * np.exp(-1j * kx * 2 * np.pi) + self.H_p0 * np.exp(
                    1j * kx * 2 * np.pi)

        return H_Matrix

    def energy(self, N, kxmin=-np.pi, kxmax=np.pi, kymin=-np.pi, kymax=np.pi):

        """
        Function calculating the energy spectrum for a given Hamiltonian
        :param N: Number of k-points
        :param kxmin: Minimum k-value in x
        :param kxmax: Maximum k-value in x
        :param kymin: Minimum k-value in y
        :param kymax: Maximum k-value in y
        :return: energy values
        """

        # Ensure that the Hamiltonian is constructed
        self.build_Hamiltonian()

        if self.lattice.typ == 'Sheet':

            # Build meshgrid for diagonalization
            kx = np.linspace(kxmin, kxmax, N + 1)
            ky = np.linspace(kymin, kymax, N + 1)
            KX, KY = np.meshgrid(kx, ky)

            # Add axis to object
            self.axis1 = KX
            self.axis2 = KY

            # Prepare solution array
            ev = []

            # For each k value, diagonalize Hamiltonian
            print('Computing bandstructure ... ', flush=True)
            if self.lattice.stack == 'Twisted Bilayer':
                for i, x in enumerate(kx):
                    for j, y in enumerate(ky):
                        ev.append(eigsh(self.get_Hamiltonian(x, y)))

            else:
                for i, x in enumerate(kx):
                    for j, y in enumerate(ky):
                        ev.append(eigvalsh(self.get_Hamiltonian(x, y)))

            ev = np.array(ev)

            ev = ev.reshape((N + 1, N + 1, len(self.lattice.orb)))

        elif self.lattice.typ == 'Ribbon':

            # k array for diagonalization
            kx = np.linspace(kxmin, kxmax, N + 1)

            # Add axis to object
            self.axis1 = kx

            # Prepare solution array
            ev = []

            # For each k value, diagonalize Hamiltonian
            print('Computing bandstructure ... ', flush=True)
            for i, x in enumerate(kx):
                ev.append(eigvalsh(self.get_Hamiltonian(x)))

            ev = np.array(ev)
            ev = ev.reshape((N + 1, len(self.lattice.orb)))

        return ev

    def dos(self, N, emin = -0.5, emax = 0.5, kxmin=-np.pi, kxmax=np.pi, kymin=-np.pi, kymax=np.pi):

        """
        Calculate density of states
        :param N: Number of k-points
        :param kxmin: Minimum k-value in x
        :param kxmax: Maximum k-value in x
        :param kymin: Minimum k-value in y
        :param kymax: Maximum k-value in y
        :return: Density of states, energies
        """

        # Ensure that the Hamiltonian is constructed
        self.build_Hamiltonian()

        if self.lattice.typ == 'Sheet':

            # Build meshgrid for diagonalization
            kx = np.linspace(kxmin, kxmax, N + 1)
            ky = np.linspace(kymin, kymax, N + 1)
            KX, KY = np.meshgrid(kx, ky)

            # Add axis to object
            self.axis1 = KX
            self.axis2 = KY

            # Prepare solution array
            ev = []

            # For each k value, diagonalize Hamiltonian
            for i, x in enumerate(kx):
                for j, y in enumerate(ky):
                    ev.append(eigvalsh(self.get_Hamiltonian(x, y)))

            ev = np.array(ev)
            ev = ev.reshape((N + 1, N + 1, len(self.lattice.orb)))

        elif self.lattice.typ == 'Ribbon':

            # k array for diagonalization
            kx = np.linspace(kxmin, kxmax, N + 1)

            # Add axis to object
            self.axis1 = kx

            # Prepare solution array
            ev = []

            # For each k value, diagonalize Hamiltonian
            for i, x in enumerate(kx):
                ev.append(eigvalsh(self.get_Hamiltonian(x)))

            ev = np.array(ev)
            ev = ev.reshape((N + 1, len(self.lattice.orb)))

        # calculate density of states from number of states with given energy
        bn = np.linspace(emin, emax, 100)
        dos = np.histogram(ev, bins=bn, range=(emin, emax))

        fo = open(os.path.join('/tmp', "Dos.TXT"), "w")

        for b in range(len(dos[0])):

            fo.write(str(dos[1][b]) + "   " + str(dos[0][b]))
            fo.write("\n")
        fo.close()


        return dos, ev


    def get_wf(self, kx, ky=0):
        """
        Function calculating eigenstates and eigenvectors for a specific k value
        :param kx: kx momentum
        :param ky: ky momentum
        :return: eigenvalues and states
        """

        # Ensure that the Hamiltonian is constructed
        self.build_Hamiltonian()
        ev, evec = eigh(self.get_Hamiltonian(kx, ky))  # Diagonalize Hamiltonian to get wavefunctions and energies

        return ev.flatten(), evec

    def current_per_site(self):

        # Ensure that the Hamiltonian is constructed
        self.build_Hamiltonian()
        j = []
        dx = 0.01
        kx = np.arange(0, 2 * np.pi / (np.sqrt(3)), dx)



        res = np.zeros((len(kx), len(self.lattice.orb), len(self.lattice.orb)), dtype='complex')

        for i in range(len(kx)):
            es, efs = self.get_wf(kx[i])  # Get wavefunction and energies

            efs = np.conjugate(efs.transpose())
            occef = []
            for (ie, iw) in zip(es, efs):  # Go through all energies and wavefunctions

                if ie < 0:  # if below fermi
                    occef.append(iw)


            op = (self.get_Hamiltonian(kx[i]+dx)-self.get_Hamiltonian(kx[i]))/dx
            for j in range(len(occef)):
                left = np.transpose(np.conjugate(occef[j]))
                right = occef[j]

                mat = np.outer(right, left)*(np.matmul(left, np.matmul(op, right)))
                j_per_site_per_state = np.diag(mat)
                res[i, j, :] = j_per_site_per_state


                #for x in kx:
        #    dH = (self.get_Hamiltonian(x+dx)-self.get_Hamiltonian(x))/dx
        #    j.append(np.diag(dH))

        #j = np.array(j)
        #j = j.reshape((len(kx), len(self.lattice.orb)))

        return res


    def current_per_site_2(self, e_low, e_high):

        # Ensure that the Hamiltonian is constructed
        self.build_Hamiltonian()

        dx = 0.01
        kx = np.arange(0, 2 * np.pi / (np.sqrt(3)), dx)



        res = np.zeros((len(kx), len(self.lattice.orb), len(self.lattice.orb)), dtype='complex')

        for i in range(len(kx)):
            es, efs = self.get_wf(kx[i])  # Get wavefunction and energies

            efs = efs.transpose()
            occef = []
            for (ie, iw) in zip(es, efs):  # Go through all energies and wavefunctions

                if ie < e_high and ie > e_low:  # if in range
                    occef.append(iw)


            op = np.divide(np.subtract(self.get_Hamiltonian(kx[i]+dx),self.get_Hamiltonian(kx[i])),dx)

            for j in range(len(occef)):
                left = np.transpose(np.conjugate(occef[j]))
                right = occef[j]

                mat = (np.matmul(left, np.matmul(op, right)))

                for l in range(len(self.lattice.orb)):
                    res[i, j, l] = mat*np.abs(right[l])**2
                #j_per_site_per_state = np.diag(mat)
                #res[i, j, :] = j_per_site_per_state




        j = np.sum(res,axis=(0, 1))

        return j


    def current_per_site_3(self, e_low, e_high):





        # Ensure that the Hamiltonian is constructed
        self.build_Hamiltonian()

        dx = 0.01
        kx = np.arange(0, 2 * np.pi / (np.sqrt(3)), dx)



        res = np.zeros((len(kx), len(self.lattice.orb), len(self.lattice.orb)), dtype='complex')

        for i in range(len(kx)):
            es, efs = self.get_wf(kx[i])  # Get wavefunction and energies

            efs = efs.transpose()
            occef = []
            for (ie, iw) in zip(es, efs):  # Go through all energies and wavefunctions

                if ie < e_high and ie > e_low:  # if in range
                    occef.append(iw)

            dhdk = 1j * self.lattice.nn * (
                        self.H_p0 * np.exp(1j * kx[i] * self.lattice.nn) - self.H_m0 * np.exp(-1j * kx[i] * self.lattice.nn))


            for j in range(len(occef)):
                left = np.transpose(np.conjugate(occef[j]))
                right = occef[j]

                mat = (np.matmul(left, np.matmul(dhdk, right)))

                for l in range(len(self.lattice.orb)):
                    res[i, j, l] = mat*np.abs(right[l])**2





        j = np.sum(res,axis=(0, 1))

        return j

    """
    def weighted_current(self, e_low, e_high, nk=400, fun=None):
        Calculate the Ground state current
        if fun is None:
            delta = 0.01

            def fun(e): return (-np.tanh(e / delta) + 1.0) / 2.0
        jgs = np.zeros(self.H_00.shape[0])  # current array
        self.build_Hamiltonian()  # generator

        ks = np.linspace(0.0, 1.0, nk, endpoint=False)  # k-points
        for k in ks:  # loop
              # Hamiltonian
            (es, ws) = self.get_wf(k)  # diagonalize
            ws = ws.transpose()  # transpose
            jk = 1j * self.lattice.nn * (self.H_p0 * np.exp(1j * k * self.lattice.nn) - self.H_m0 * np.exp(-1j * k * self.lattice.nn))
            #jk = np.identity(len(ws))
            for (e, w) in zip(es, ws):
                if e < e_high and e > e_low:
                    weight = fun(e)
                    #print(weight)
                    we = np.matrix(w)
                    d = ((we.T).H * jk * we.T)[0, 0]
                    #d = np.conjugate(w) * ket_Aw(jk, w)  # current density
                    jgs += d.real   # add contribution
                    jgs += (np.abs(w)**2*weight).real # add contribution
        jgs /= nk  # normalize
        print("Total current", np.sum(jgs))
        np.savetxt("CURRENT1D.OUT", np.matrix([range(len(jgs)), jgs]).T)

        return jgs
    """

    def expec_operators(self, es=np.linspace(-1.0, 1.0, 100), delta=0.01, nk=100, op=None):
        """
        Function computing expectation values of operators
        :param es: energy window
        :param delta: delta for
        :param nk: number of k points
        :param op: operator for expectation value
        :return: solution array
        """
        print("Calculating eigenvectors")
        ps = []  # weights
        evals, ws = [], []  # empty list
        ks = np.linspace(0, 2 * np.pi / (np.sqrt(3)), nk)  # get grid

        jk = lambda k: 1j * self.lattice.nn * (self.H_p0 * np.exp(1j * k * self.lattice.nn) - self.H_m0 * np.exp(-1j * k * self.lattice.nn)) # current operator

        if op is "current": op = lambda x,k: np.matmul(np.conjugate(x), np.matmul(jk(k), x)).real # current operator
        elif op is None: op = lambda x,k: 1.0 # operator for ldos

        print("Diagonalizing")
        for k in ks:  # loop
            e, w = eigh(self.get_Hamiltonian(k))
            w = w.transpose()
            evals += [ie for ie in e]
            ws += [iw for iw in w]

            ps += [op(iw, k=k) for iw in w]  # weights

        ds = [(np.conjugate(v) * v).real for v in ws]  # calculate densities
        del ws  # remove the wavefunctions



        fo = open(os.path.join('/tmp', "LDOS.TXT"), "w")  # files with the results

        def getldosi(e):
            """Get this iteration"""
            out = np.array([0.0 for i in range(self.H_00.shape[0])])  # initialize
            for (d, p, ie) in zip(ds, ps, evals):  # loop over wavefunctions
                fac = delta / ((e - ie) ** 2 + delta ** 2)  # factor to create a delta
                out += fac * d * p  # add contribution
            out /= np.pi  # normalize
            return out  # resum if necessary


        ie = 0
        ldos = []
        for e in es:  # loop over energies
            print("Ldos/Current for energy", e)
            out = getldosi(e)
            ldos.append(out)

            ind = 0
            for (iy, idos) in zip(self.lattice.orb[:,1], out):
                #if iz != 0:
                fo.write(str(ind) + "   " + str(e) + "   " + str(idos))
                fo.write("\n")# name of the file
                ind += 1

        fo.close()  # close file

        return ldos


