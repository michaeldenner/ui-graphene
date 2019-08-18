import numpy as np

"""
This file contains the Hamiltonians to be used for numerical diagonalization
"""

def f(kx, ky, t1, t2, t3, a=1):
    """
    This is the intra layer phase factor
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param a: lattice constant, default = 1
    :param t1: hopping within unit cell A-B
    :param t2: hopping between unit cells
    :param t3: hopping between unit cells
    :return: phase factor evaluated at meshgrid momenta
    """


    return t1 * np.exp(1j*(ky*(a))) + t2 * np.exp(1j*(ky*(-a/2) + kx*(np.sqrt(3)* a/2))) + t3 * np.exp(1j*(ky*(-a/2) + kx*(-np.sqrt(3)* a/2)))


def f_phi(kx, ky, t1, t2, t3, phi, a=1):
    """
    This is the intra layer phase factor
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param t1: hopping within unit cell A-B
    :param t2: hopping between unit cells
    :param t3: hopping between unit cells
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: phase factor evaluated at meshgrid momenta
    """


    return t1*np.exp(1j*phi)*np.exp(1j*(ky*(a))) + t2 * np.exp(-1j*(phi/2))* np.exp(1j*(ky*(-a/2) + kx*(np.sqrt(3)* a/2))) + t3 * np.exp(-1j*(phi/2)) * np.exp(1j*(ky*(-a/2) + kx*(-np.sqrt(3)* a/2)))


def f_angle(kx, ky, t1, t2, t3, phi, theta, a=1):
    """
    Intra layer phase factor for a magnetic field at an angle
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param t1: hopping within unit cell A-B
    :param t2: hopping between unit cells
    :param t3: hopping between unit cells
    :param phi: External field strength
    :param theta: Angle of field
    :param a: lattice constant, default = 1
    :return: phase factor evaluated at meshgrid momenta
    """

    return t1*np.exp(1j*phi*np.sin(theta))*np.exp(1j*(ky*(a))) + t2 * np.exp(-1j*((phi*(np.sin(theta)+np.sqrt(3)*np.cos(theta)))/2))* np.exp(1j*(ky*(-a/2) + kx*(np.sqrt(3)* a/2))) + t3 * np.exp(-1j*((phi*(np.sin(theta)-np.sqrt(3)*np.cos(theta)))/2)) * np.exp(1j*(ky*(-a/2) + kx*(-np.sqrt(3)* a/2)))


def F(kx, ky, a=1):
    """
    Intra layer phase factor for nnn
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param a: lattice constant, default = 1
    :return: phase factor evaluated at meshgrid momenta
    """
    return 2*np.cos(ky*(np.sqrt(3)* a)) + 4*np.cos(ky*(np.sqrt(3)* a/2))*np.cos(3*kx*a/2)


def ftilde_phi(kx, ky, phi, a=1):
    """
    Inter layer phase factor for a magnetic field in nnn
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: phase factor evaluated at meshgrid momenta
    """

    return np.exp(1j*(ky*(a)))*np.exp(1j*phi/2) + np.exp(-1j*phi/4)* np.exp(1j*(ky*(-a/2) + kx*(np.sqrt(3)* a/2))) +  np.exp(-1j*phi/4)*np.exp(1j*(ky*(-a/2) + kx*(-np.sqrt(3)* a/2)))

def F_phi(kx, ky, phi, a=1):
    """
    Intra layer phase factor for a magnetic field in nnn
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: phase factor evaluated at meshgrid momenta
    """

    return 2*np.cos(ky*(np.sqrt(3)* a)) + 4*np.cos(ky*(np.sqrt(3)* a/2))*np.cos((3/2)*(kx*a+phi))

def f_Landau(kx, ky, t1, t2, t3, phi, a=1):
    """
    This is the intra layer phase factor
    :param kx: meshgrid of kx values
    :param ky: meshgrid of ky values
    :param t1: hopping within unit cell A-B
    :param t2: hopping between unit cells
    :param t3: hopping between unit cells
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: phase factor evaluated at meshgrid momenta
    """


    return t1*np.exp(1j*phi*a)*np.exp(1j*(ky*(a))) + t2 * np.exp(-1j*(np.sqrt(3)/4)*phi*a) * np.exp(1j*((ky + (np.sqrt(3)/2)*phi) *(-a/2) + (kx - phi/2)*(np.sqrt(3)* a/2))) + t3 * np.exp(1j*(np.sqrt(3)/4)*phi*a) * np.exp(1j*((ky - (np.sqrt(3)/2)*phi)*(-a/2) + (kx - phi/2)*(-np.sqrt(3)* a/2)))


def Monolayer(KX,KY,N, t1, t2, t3, a = 1):
    """
    This is the Hamiltonian for monolayer graphene
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param a: lattice constant, default = 1
    :param t1: hopping within unit cell A-B
    :param t2: hopping between unit cells
    :param t3: hopping between unit cells
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus = np.array([[0, 1], [0, 0]])
    pauli_minus = np.array([[0, 0], [1, 0]])

    # Intra layer hopping
    a1 = np.kron(f(KX, KY, t1, t2, t3, a), pauli_plus)
    a2 = np.kron(np.conjugate(f(KX, KY, t1, t2, t3, a)), pauli_minus)
    intra_layer_1 = a1 + a2

    # Construct full Hamiltonian
    H = intra_layer_1
    H = np.reshape(H, (N + 1, 2, N + 1, 2))
    H = np.transpose(H, (0, 2, 1, 3))

    return H


def h_0(kx, ky, t1, b, phi):
    k = np.array([kx, ky])
    b = np.array(b)
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]
    h0 = np.cos(kx * b1[0] + ky * b1[1]) + np.cos(kx * b2[0] + ky * b2[1]) + np.cos(kx * b3[0] + ky * b3[1])

    return h0 * np.cos(phi) * 2 * t1


def h_1(kx, ky, t, a):
    k = np.array([kx, ky])
    a = np.array(a)
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    h1 = np.cos(kx * a1[0] + ky * a1[1]) + np.cos(kx * a2[0] + ky * a2[1]) + np.cos(kx * a3[0] + ky * a3[1])

    return h1 * t


def h_2(kx, ky, t, a):
    k = np.array([kx, ky])
    a = np.array(a)
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    h2 = np.sin(kx * a1[0] + ky * a1[1]) + np.sin(kx * a2[0] + ky * a2[1]) + np.sin(kx * a3[0] + ky * a3[1])

    return h2 * t


def h_3(kx, ky, t1, b, phi, V):
    k = np.array([kx, ky])
    b = np.array(b)

    b1 = b[0]
    b2 = b[1]
    b3 = b[2]
    h3 = np.sin(kx * b1[0] + ky * b1[1]) + np.sin(kx * b2[0] + ky * b2[1]) + np.sin(kx * b3[0] + ky * b3[1])

    return V + h3 * np.sin(phi) * 2 * t1


def Haldane(kx, ky, N, t, t1, phi, V):
    b = np.array([[0, np.sqrt(3)], [3 / 2, np.sqrt(3) / 2], [3 / 2, -np.sqrt(3) / 2]])
    a = np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2], [-1 / 2, -np.sqrt(3) / 2]])

    one = np.array([[1, 0], [0, 1]])
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    H0 = np.kron(h_0(kx, ky, t1, b, phi), one)
    H1 = np.kron(h_1(kx, ky, t, a), pauli_x)
    H2 = np.kron(h_2(kx, ky, t, a), pauli_y)
    H3 = np.kron(h_3(kx, ky, t1, b, phi, V), pauli_z)

    H = H0 + H1 + H2 + H3
    H = np.reshape(H, (N + 1, 2, N + 1, 2))
    H = np.transpose(H, (0, 2, 1, 3))

    return H


def Bilayer_origin_A2(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7, a = 1):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param a: lattice constant, default = 1
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])

    """
    Hamiltonian components
    """

    # Intra layer 1 hopping
    a1 = np.kron(f(KX, KY, t1, t2, t3, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f(KX, KY, t1, t2, t3, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H


def Bilayer_origin_A2_AB(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, a = 1):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param a: lattice constant, default = 1
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])


    """
    Hamiltonian components
    """

    # Intra layer 1 hopping
    a1 = np.kron(f(KX, KY, t1, t2, t3, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f(KX, KY, t1, t2, t3, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_plus_2)
    b2 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)


    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Bilayer_origin_A2_field(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7,phi, a = 1, V = 0):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2 in an external field along x
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    upper_layer = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    lower_layer = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])

    """
    Hamiltonian components
    """

    # Intra layer 1 hopping
    a1 = np.kron(f_phi(KX, KY, t1, t2, t3, phi, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_phi(KX, KY, t1, t2, t3, phi, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Onside energy

    bias_up = np.kron(V * np.exp(0 * KX + 0 * KY), upper_layer)
    bias_down = np.kron(-V * np.exp(0 * KX + 0 * KY), lower_layer)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + bias_down + bias_up
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Bilayer_Landau(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7,phi, a = 1):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2 in an external field along x
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])

    """
    Hamiltonian components
    """

    # Intra layer 1 hopping
    a1 = np.kron(f_Landau(KX, KY, t1, t2, t3, phi, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_Landau(KX, KY, t1, t2, t3, phi, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Bilayer_origin_A2_field_AB(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6,phi, a = 1):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2 in an external field along x
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])


    """
    Hamiltonian components
    """

    # Intra layer 1 hopping
    a1 = np.kron(f_phi(KX, KY, t1, t2, t3, phi, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_phi(KX, KY, t1, t2, t3, phi, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)


    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Bilayer_AA_effective(KX, KY, N, t1, phi, a = 1):
    """
    Effective Hamiltonian around semi Dirac cone in AA stacked graphene in an external field
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """



    """
    Matrices
    """

    pauli_plus = np.array([[0, 1], [0, 0]])
    pauli_minus = np.array([[0, 0], [1, 0]])

    """
    Hamiltonian components
    """

    b1 = np.kron(1.5 * t1 * a * (KX - 1j * KY - 1j * phi * KX), pauli_plus)
    b2 = np.kron(np.conjugate(1.5 * t1 * a * (KX - 1j * KY - 1j * phi * KX)), pauli_minus)
    effective = b1 + b2

    # Construct full Hamiltonian
    H = effective
    H = np.reshape(H, (N + 1, 2, N + 1, 2))
    H = np.transpose(H, (0, 2, 1, 3))

    return H


def Bilayer_origin_A2_field_angle(KX, KY, N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7, phi, theta, a=1, V = 0):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2 in an external field at an angle
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param phi: External field strength
    :param theta: Angle of magnetic field
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    upper_layer = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    lower_layer = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])



    """
    Hamiltonian
    """

    # Intra layer 1 hopping
    a1 = np.kron(f_angle(KX, KY, t1, t2, t3, phi, theta, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_angle(KX, KY, t1, t2, t3, phi, theta, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Onside energy

    bias_up = np.kron(V * np.exp(0 * KX + 0 * KY), upper_layer)
    bias_down = np.kron(-V * np.exp(0 * KX + 0 * KY), lower_layer)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + bias_up + bias_down
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))



    return H

def Bilayer_origin_A2_field_angle_trueAB(KX, KY, N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7, phi, theta, a=1, eps_up = 0, eps_down = 0):
    """
    Bilayer Hamiltonian for phase setting with origin at lattice site A2 in an external field at an angle
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t2: hopping between unit cells layer 1
    :param t3: hopping between unit cells layer 1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t2_tilde: hopping between unit cells layer 2
    :param t3_tilde: hopping between unit cells layer 2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param phi: External field strength
    :param theta: Angle of magnetic field
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    upper_layer = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    lower_layer = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])



    """
    Hamiltonian
    """

    # Intra layer 1 hopping
    a1 = np.kron(f_angle(KX, KY, t1, t2, t3, phi, theta, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_angle(KX, KY, t1, t2, t3, phi, theta, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_minus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_plus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Onside energy

    bias_up = np.kron(eps_up * np.exp(0 * KX + 0 * KY), upper_layer)
    bias_down = np.kron(eps_down * np.exp(0 * KX + 0 * KY), lower_layer)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + bias_up + bias_down
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))



    return H


def Bilayer_origin_A2_NNN(KX, KY, N, t1, t1_tilde, t6, t7, t_prime, t_prime_prime, t10, t11, a=1):
    """
    Next nearest neighbor bilayer Hamiltonian (origin at A2)
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param t_prime: nnn hopping in layer 1
    :param t_prime_prime: nnn hopping in layer 2
    :param t10: nnn hopping between layers
    :param t11: nnn hopping between layers
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
    one_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    one_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])

    nnn_1 = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    nnn_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    nnn_3 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    nnn_4 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    """
    Hamiltonian
    """

    # Intra layer 1 hopping
    a1 = np.kron(f(KX, KY, t1, t1, t1, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f(KX, KY, t1, t1, t1, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t1_tilde, t1_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t1_tilde, t1_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(-t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(-t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Inter layer nnn hopping

    nnn_inter_layer_1 = np.kron(-f(KX, KY, t10, t10, t10, a), nnn_1)
    nnn_inter_layer_2 = np.kron(-np.conjugate(f(KX, KY, t11, t11, t11, a)), nnn_2)
    nnn_inter_layer_3 = np.kron(-f(KX, KY, t11, t11, t11, a), nnn_3)
    nnn_inter_layer_4 = np.kron(-np.conjugate(f(KX, KY, t10, t10, t10, a)), nnn_4)

    # NNN hopping intra layer
    nnn_intra_layer_1 = np.kron(t_prime * F(KX, KY, a), one_1)
    nnn_intra_layer_2 = np.kron(t_prime_prime * F(KX, KY, a), one_2)

    nnn = nnn_inter_layer_1 + nnn_inter_layer_2 + nnn_inter_layer_3 + nnn_inter_layer_4 + nnn_intra_layer_1 + nnn_intra_layer_2

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + nnn
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H



def Bilayer_origin_A2_NNN_field(KX, KY, N, t1, t1_tilde, t6, t7, t_prime, t_prime_prime, t10, t11, phi, a=1):
    """
    Next nearest neighbor bilayer Hamiltonian (origin at A2) with field along x
    :param KX: meshgrid of kx values
    :param KY: meshgrid of ky values
    :param N: Number of mesh points
    :param t1: hopping within unit cell A1-B1
    :param t1_tilde: hopping within unit cell A2-B2
    :param t6: hopping within unit cell A1-A2
    :param t7: hopping within unit cell B1-B2
    :param t_prime: nnn hopping in layer 1
    :param t_prime_prime: nnn hopping in layer 2
    :param t10: nnn hopping between layers
    :param t11: nnn hopping between layers
    :param phi: External field strength
    :param a: lattice constant, default = 1
    :return: Hamiltonian evaluated at meshgrid momenta
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
    one_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    one_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])

    nnn_1 = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    nnn_2 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    nnn_3 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    nnn_4 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    """
    Hamiltonian
    """

    # Intra layer 1 hopping
    a1 = np.kron(f_phi(KX, KY, t1, t1, t1, phi, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_phi(KX, KY, t1, t1, t1, phi, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t1_tilde, t1_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t1_tilde, t1_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(-t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(-t7 * np.exp(0 * KX + 0 * KY), m_t7)

    # Inter layer nnn hopping

    nnn_inter_layer_1 = np.kron(-t10 * ftilde_phi(KX, KY, phi, a), nnn_1)
    nnn_inter_layer_2 = np.kron(-t11 * np.conjugate(ftilde_phi(KX, KY, phi, a)), nnn_2)
    nnn_inter_layer_3 = np.kron(-t11 * ftilde_phi(KX, KY, phi, a), nnn_3)
    nnn_inter_layer_4 = np.kron(-t10 * np.conjugate(ftilde_phi(KX, KY, phi, a)), nnn_4)


    # NNN hopping intra layer
    nnn_intra_layer_1 = np.kron(t_prime * F_phi(KX, KY, phi, a), one_1)
    nnn_intra_layer_2 = np.kron(t_prime_prime * F(KX, KY, a), one_2)

    nnn = nnn_inter_layer_1 + nnn_inter_layer_2 + nnn_inter_layer_3 + nnn_inter_layer_4 + nnn_intra_layer_1 + nnn_intra_layer_2

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + nnn
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Bilayer_artificial(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7, t8, t9, a = 1):
    """
    This artificial bilayer Hamiltonian should produce semi Dirac points
    :param KX:
    :param KY:
    :param N:
    :param t1:
    :param t2:
    :param t3:
    :param t1_tilde:
    :param t2_tilde:
    :param t3_tilde:
    :param t6:
    :param t7:
    :param a:
    :return:
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
    m_t8 = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    m_t9 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

    """
    Hamiltonian components
    """

    # Intra layer 1 hopping
    a1 = np.kron(f(KX, KY, t1, t2, t3, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f(KX, KY, t1, t2, t3, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)
    inter_layer_t8 = np.kron(t8 * np.exp(0 * KX + 0 * KY), m_t8)
    inter_layer_t9 = np.kron(t9 * np.exp(0 * KX + 0 * KY), m_t9)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + inter_layer_t8 + inter_layer_t9
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Bilayer_artificial_field(KX,KY,N, t1, t2, t3, t1_tilde, t2_tilde, t3_tilde, t6, t7, t8, t9,phi, a = 1):
    """
    This artificial bilayer Hamiltonian should produce semi Dirac points
    :param KX:
    :param KY:
    :param N:
    :param t1:
    :param t2:
    :param t3:
    :param t1_tilde:
    :param t2_tilde:
    :param t3_tilde:
    :param t6:
    :param t7:
    :param a:
    :return:
    """

    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
    m_t8 = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    m_t9 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

    """
    Hamiltonian components
    """
    # Intra layer 1 hopping
    a1 = np.kron(f_phi(KX, KY, t1, t2, t3, phi, a), pauli_plus_1)
    a2 = np.kron(np.conjugate(f_phi(KX, KY, t1, t2, t3, phi, a)), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a), pauli_plus_2)
    b2 = np.kron(np.conjugate(f(KX, KY, t1_tilde, t2_tilde, t3_tilde, a)), pauli_minus_2)
    intra_layer_2 = b1 + b2


    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * KX + 0 * KY), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * KX + 0 * KY), m_t7)
    inter_layer_t8 = np.kron(t8 * np.exp(0 * KX + 0 * KY), m_t8)
    inter_layer_t9 = np.kron(t9 * np.exp(0 * KX + 0 * KY), m_t9)

    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7 + inter_layer_t8 + inter_layer_t9
    H = np.reshape(H, (N + 1, 4, N + 1, 4))
    H = np.transpose(H, (0, 2, 1, 3))

    return H

def Dimer_chain(ky,N,t,t6,t7,phi,a=1):
    """
    Matrices
    """

    pauli_plus_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_minus_1 = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pauli_plus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    pauli_minus_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])

    m_t6 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    m_t7 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]])


    """
    Hamiltonian components
    """
    # Intra layer 1 hopping
    a1 = np.kron(t*2*np.cos(ky*a-phi), pauli_plus_1)
    a2 = np.kron(t*2*np.cos(ky*a-phi), pauli_minus_1)
    intra_layer_1 = a1 + a2

    # Intra layer 2 hopping
    b1 = np.kron(t*2*np.cos(ky*a), pauli_plus_2)
    b2 = np.kron(t*2*np.cos(ky*a), pauli_minus_2)
    intra_layer_2 = b1 + b2

    # Inter layer hopping
    inter_layer_t6 = np.kron(t6 * np.exp(0 * ky), m_t6)
    inter_layer_t7 = np.kron(t7 * np.exp(0 * ky), m_t7)


    # Construct full Hamiltonian
    H = intra_layer_1 + intra_layer_2 + inter_layer_t6 + inter_layer_t7

    H = np.reshape(H, (4, N + 1, 4))
    H = np.transpose(H, (1, 0, 2))


    return H
