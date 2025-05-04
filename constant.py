import numpy as np
markers = (
    ".",   # Point marker
    ",",   # Pixel marker
    "o",   # Circle marker
    "v",   # Triangle down marker
    "^",   # Triangle up marker
    "<",   # Triangle left marker
    ">",   # Triangle right marker
    "1",   # Tri-down marker
    "2",   # Tri-up marker
    "3",   # Tri-left marker
    "4",   # Tri-right marker
    "s",   # Square marker
    "p",   # Pentagon marker
    "*",   # Star marker
    "h",   # Hexagon1 marker
    "H",   # Hexagon2 marker
    "+",   # Plus marker
    "x",   # X marker
    "D",   # Diamond marker
    "d",   # Thin diamond marker
    "|",   # Vertical line marker
    "_",   # Horizontal line marker
)
# Physical layer constants
q = 1.602e-19  # Charge of an electron (Coulombs)
k_B = 1.381e-23  # Boltzmann constant (J/K)
Temperature = 290  # Temperature in Kelvin (room temperature)
R_L = 50  # Load resistance in Ohms

# WOC parameters
IlluminationCoefficient = 0.3
f_WOC = 100e9
B_WOC = 1e9
P_tx_WOC = 5
T_optical = 0.99
R = 0.99
R_photo = 0.6
lamda = 3e8 / f_WOC
u = 1
theta = phi = 60
A_rec = 0.0023
P_N = 10**(-21)
V_bias = 3

# RF parameters
f_RF = 2.4e9
lambda_RF = 3e8 / f_RF
B_RF = 20e6
P_tx_RF = 0.05
attenuation_coefficient_RF = 3
d0 = 1
PL0 = 20 * np.log10(4 * np.pi * d0 / lambda_RF)
sigma_shadowing = 10
G_tx_RF = 1
G_rx_RF = 1 
transferTime=1 #time for transfer of one packet