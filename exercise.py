import numpy as np
import matplotlib.pyplot as plt


phi_plus = np.array([1,0,0,1])/np.sqrt(2)
phi_minus = np.array([1,0,0,-1])/np.sqrt(2)
psi_plus = np.array([0,1,1,0])/np.sqrt(2)
psi_minus = np.array([0,1,-1,0])/np.sqrt(2)

def get_coefs(state, theta, phi):
    # define the basis vectors
    e1 = np.array([
        [np.cos(theta)],
        [np.exp(1j*phi)*np.sin(theta)]])
    e2 = np.array([
        [np.sin(theta)],
        [-np.exp(1j*phi)*np.cos(theta)]])

    # define the two-qubit basis
    e1e1 = np.kron(e1,e1)
    e1e2 = np.kron(e1,e2)
    e2e1 = np.kron(e2,e1)
    e2e2 = np.kron(e2,e2)

    # obtain the coefficients in the new basis
    return np.array([np.vdot(e, state) for e in [e1e1,e1e2,e2e1,e2e2]])



