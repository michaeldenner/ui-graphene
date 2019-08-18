import numpy as np
import os ; import sys ;
sys.path.append(os.environ['PYGRAROOT'])





class lattice:

    """
    This class generates a lattice of specified geometry
    """
    def __init__(self, typ, stack, nunit=1, twistangle = 6):
        """
        Initialize object
        :param typ: lattice type
        :param stack: stacking type
        :param nunit: number of unit cells
        """
        self.typ = typ
        self.stack = stack
        self.nunit = nunit
        self.twistangle = twistangle

        if self.typ == 'Ribbon':

            if self.stack == 'AC':

                # Define lattice vectors
                self.lat = np.array([3., 0])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0.], [1., 0.], [-1. / 2., np.sqrt(3) / 2.], [3. / 2., np.sqrt(3) / 2.]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0, np.sqrt(3)]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 2))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 1

            elif self.stack == 'AC AA':

                # Define lattice vectors
                self.lat = np.array([3., 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [0., 0., np.sqrt(2)], [1., 0., 0.], [1., 0., np.sqrt(2)],
                                     [-1. / 2., np.sqrt(3) / 2., 0.], [-1. / 2., np.sqrt(3) / 2., np.sqrt(2)],
                                     [3. / 2., np.sqrt(3) / 2., 0.], [3. / 2., np.sqrt(3) / 2., np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., np.sqrt(3), 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 2

            elif self.stack == 'AC AB':

                # Define lattice vectors
                self.lat = np.array([3., 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [-1., 0., np.sqrt(2)], [1., 0., 0.], [0., 0., np.sqrt(2)],
                                     [-1. / 2., np.sqrt(3) / 2., 0.], [-3. / 2., np.sqrt(3) / 2., np.sqrt(2)],
                                     [3. / 2., np.sqrt(3) / 2., 0.], [1. / 2., np.sqrt(3) / 2., np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., np.sqrt(3), 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 2


            elif self.stack == 'ZZ':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0.], [-np.sqrt(3) / 2., 1. / 2.], [-np.sqrt(3) / 2., 3. / 2.], [0., 2.]])

                sub_unit = np.array([0, 1, 0, 1])

                # Prepare array for all orbitals
                orb = []
                suborb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3.]))
                    suborb.append(sub_unit)

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 2))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                suborb = np.array(suborb)
                suborb.flatten()
                suborb = np.reshape(suborb, (nunit * len(orb_unit)))

                self.orb = orb
                self.sublattice = suborb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 1

            elif self.stack == 'Haldane':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0.], [-np.sqrt(3) / 2., 1. / 2.], [-np.sqrt(3) / 2., 3. / 2.], [0., 2.]])

                sub_unit = np.array([0, 1, 0, 1])

                # Prepare array for all orbitals
                orb = []
                suborb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3.]))
                    suborb.append(sub_unit)

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 2))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                suborb = np.array(suborb)
                suborb.flatten()
                suborb = np.reshape(suborb, (nunit * len(orb_unit)))

                self.orb = orb
                self.sublattice = suborb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 1

            elif self.stack == 'ZZ AA':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [0., 0., np.sqrt(2)], [-np.sqrt(3) / 2., 1. / 2., 0.],
                                     [-np.sqrt(3) / 2., 1. / 2., np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 0.],
                                     [-np.sqrt(3) / 2., 3. / 2., np.sqrt(2)], [0., 2., 0.], [0., 2., np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3., 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 2

            elif self.stack == 'ZZ AB':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [0., 1., np.sqrt(2)], [-np.sqrt(3) / 2., 1. / 2., 0.],
                                     [-np.sqrt(3) / 2., 3. / 2., np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 0.],
                                     [-np.sqrt(3) / 2., 5. / 2., np.sqrt(2)], [0., 2., 0.], [0., 3., np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3., 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 2

            elif self.stack == 'ZZ ABA':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [0., 1., np.sqrt(2)],[0., 0., 2.*np.sqrt(2)], [-np.sqrt(3) / 2., 1. / 2., 0.],
                                     [-np.sqrt(3) / 2., 1. / 2., 2.*np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 2.*np.sqrt(2)],
                                     [-np.sqrt(3) / 2., 3. / 2., np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 0.],
                                     [-np.sqrt(3) / 2., 5. / 2., np.sqrt(2)], [0., 2., 0.], [0., 3., np.sqrt(2)], [0., 2., 2.*np.sqrt(2)],])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3., 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 3

            elif self.stack == 'ZZ ABB':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [0., 1., np.sqrt(2)], [-np.sqrt(3) / 2., 1. / 2., 0.],
                                     [-np.sqrt(3) / 2., 3. / 2., np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 0.],
                                     [-np.sqrt(3) / 2., 5. / 2., np.sqrt(2)], [0., 2., 0.], [0., 3., np.sqrt(2)],
                                     [0., 1., 2*np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 2*np.sqrt(2)], [-np.sqrt(3) / 2., 5. / 2., 2*np.sqrt(2)],
                                     [0., 3., 2*np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3., 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 3

            elif self.stack == 'ZZ ABC':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., 0.], [0., 1., np.sqrt(2)], [-np.sqrt(3) / 2., 1. / 2., 0.],
                                     [-np.sqrt(3) / 2., 3. / 2., np.sqrt(2)], [-np.sqrt(3) / 2., 3. / 2., 0.],
                                     [-np.sqrt(3) / 2., 5. / 2., np.sqrt(2)], [0., 2., 0.], [0., 3., np.sqrt(2)],
                                     [0., 2., 2*np.sqrt(2)], [-np.sqrt(3) / 2., 5. / 2., 2*np.sqrt(2)], [-np.sqrt(3) / 2., 7. / 2., 2*np.sqrt(2)],
                                     [0., 4., 2*np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3., 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 3

            elif self.stack == 'ZZ AB fully centered':

                # Define lattice vectors
                self.lat = np.array([np.sqrt(3), 0., 0.])

                # Define coordinate of orbitals in unit cell
                orb_unit = np.array([[0., 0., -np.sqrt(2)/2], [0., 1., np.sqrt(2)/2], [-np.sqrt(3) / 2., 1. / 2., -np.sqrt(2)/2],
                                     [-np.sqrt(3) / 2., 3. / 2., np.sqrt(2)/2], [-np.sqrt(3) / 2., 3. / 2., -np.sqrt(2)/2],
                                     [-np.sqrt(3) / 2., 5. / 2., np.sqrt(2)/2], [0., 2., -np.sqrt(2)/2], [0., 3., np.sqrt(2)/2]])

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array([0., 3., 0.]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.sqrt(3) * 0.99
                self.layer = 2

            elif self.stack == 'Twisted Bilayer':
                from pygra import specialgeometry
                g = specialgeometry.twisted_bilayer(self.twistangle)  # geometry of 2D TBG
                self.lat = np.array([g.a2])

                orb_unit = np.array(g.r)

                # Prepare array for all orbitals
                orb = []

                # Add orbitals for given number of unit cells
                for i in range(self.nunit):
                    orb.append(orb_unit + i * np.array(g.a1))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (nunit * len(orb_unit), 3))

                # Centering around zero
                orb[:, 1] = orb[:, 1] - orb[-1, 1] / 2

                self.orb = orb
                self.nn = np.linalg.norm(np.array([g.a2]))
                self.layer = 2

            else:
                print('Stacking not supported')

        elif self.typ == 'Sheet':

            if self.stack == 'Monolayer':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5], [np.sqrt(3.0) / 2.0, 1.5]])

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0.], [0., 1.]])
                self.nn = np.sqrt(3)
                self.layer = 1

            elif self.stack == 'Bilayer AA':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5, 0.], [np.sqrt(3.0) / 2.0, 1.5, 0.]])
                self.nn = np.sqrt(3)
                self.layer = 2

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0., 0.],  [0., 0., np.sqrt(2)],[0., 1., 0.], [0., 1., np.sqrt(2)]])

            elif self.stack == 'Bilayer AB':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5, 0.], [np.sqrt(3.0) / 2.0, 1.5, 0.]])
                self.nn = np.sqrt(3)
                self.layer = 2

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0., 0.],  [0., 0., np.sqrt(2)], [0., 1., 0.], [0., -1., np.sqrt(2)]])

            elif self.stack == 'Twisted Bilayer':
                from pygra import specialgeometry
                g = specialgeometry.twisted_bilayer(self.twistangle)  # geometry of 2D TBG
                self.lat = np.array([g.a1, g.a2])


                self.orb = np.array(g.r)
                #self.nn = np.linalg.norm(np.array([g.a2]))
                self.layer = 2

            elif self.stack == 'Trilayer ABA':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5, 0.], [np.sqrt(3.0) / 2.0, 1.5, 0.]])
                self.nn = np.sqrt(3)
                self.layer = 3

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0., 0.],  [0., 0., np.sqrt(2)], [0., 0., 2*np.sqrt(2)], [0., 1., 0.], [0., -1., np.sqrt(2)], [0., 1., 2*np.sqrt(2)]])

            elif self.stack == 'Trilayer ABB':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5, 0.], [np.sqrt(3.0) / 2.0, 1.5, 0.]])
                self.nn = np.sqrt(3)
                self.layer = 3

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0., 0.],  [0., 0., np.sqrt(2)], [0., 0., 2*np.sqrt(2)], [0., 1., 0.], [0., -1., np.sqrt(2)], [0., -1., 2*np.sqrt(2)]])

            elif self.stack == 'Trilayer ABC':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5, 0.], [np.sqrt(3.0) / 2.0, 1.5, 0.]])
                self.nn = np.sqrt(3)
                self.layer = 3

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0., 0.],  [0., 0., np.sqrt(2)], [0., -1., 2*np.sqrt(2)], [0., 1., 0.], [0., -1., np.sqrt(2)], [0., -2., 2*np.sqrt(2)]])

            elif self.stack == 'Film ABC':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5, 0.], [np.sqrt(3.0) / 2.0, 1.5, 0.]])
                self.nn = np.sqrt(3)
                self.layer = 3*self.nunit

                # Define coordinates of orbitals
                orb_unit = np.array([[0., 0., 0.],  [0., 1., 0], [0., 1., np.sqrt(2)], [0., 2., np.sqrt(2)], [0., 2., 2*np.sqrt(2)], [0., 3., 2*np.sqrt(2)]])

                # Prepare array for all orbitals
                orb = []
                if self.nunit%3 == 0:
                    num = int(self.nunit//3)
                else:
                    num = int(self.nunit//3) + 1

                # Add orbitals for given number of unit cells
                for i in range(num):
                    orb.append(orb_unit + i * np.array([0., 3., 3.*np.sqrt(2)]))

                orb = np.array(orb)

                orb.flatten()
                orb = np.reshape(orb.flatten(), (num * len(orb_unit), 3))

                if self.nunit%3 == 1:
                    orb = np.delete(orb, -1, 0)
                    orb = np.delete(orb, -1, 0)
                    orb = np.delete(orb, -1, 0)
                    orb = np.delete(orb, -1, 0)
                elif self.nunit%3 == 2:
                    orb = np.delete(orb, -1, 0)
                    orb = np.delete(orb, -1, 0)

                # Centering around zero
                orb[:, 2] = orb[:, 2] - orb[-1, 2] / 2

                self.orb = orb


            elif self.stack == 'Haldane':

                # Define lattice vectors
                self.lat = np.array([[-np.sqrt(3.0) / 2.0, 1.5], [np.sqrt(3.0) / 2.0, 1.5], [np.sqrt(3), 0.0]])

                # Define coordinates of orbitals
                self.orb = np.array([[0., 0.], [0., 1.]])
                self.sublattice = np.array([0, 1])
                self.nn = np.sqrt(3)
                self.layer = 1
            else:
                print('Stacking not supported')
        else:
            print('Lattice type not supported')
