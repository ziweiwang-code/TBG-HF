import numpy as np

# initialize useful matrices/variables
omega = np.exp(2 * np.pi * 1j / 3)
omegac = np.exp(-2 * np.pi * 1j / 3)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
syc = np.array([[0, 1j], [-1j, 0]])
sz = np.array([[1, 0], [0, -1]])
s0 = np. array([[1,0],[0, 1]]) 
bG1 = 1/2*np.array([3,np.sqrt(3)]) #monolayer RLV, with kD scaled out
bG2 = 1/2*np.array([3,-np.sqrt(3)])

# fundamental constants
CCa = 1.42 * 10 ** -10  # C-C distance in m
c0 = 3.35 * 10 ** -10  # interlayer spacing
vkD = 9.905  # v_0 * hbar * k_dirac in eV
kD = 4 * np.pi / (3 * np.sqrt(3) * CCa) # k_dirac in m^-1
vhbar = vkD/kD # v_0 * hbar in eV m
hbar = 1.054571817 * 10 ** -34 # hbar in SI
echarge = 1.602176634 * 10 ** -19 # electron charge in SI
epsilon0 = 8.854 * 10 ** -12 # vacuum permittivity in SI
Poisson = 0.16 #Poisson ratio of graphene
Beta = 3.14 #strain hopping parameter