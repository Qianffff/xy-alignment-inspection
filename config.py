import numpy as np

# ===============================
# Constants
# ===============================
m = 9.11e-31            # Electron mass (kg)
e = 1.602e-19           # Elementary charge (C)
epsilon_0 = 8.854e-12   # Vacuum permittivity (F/m)
h = 6.626e-34           # Planck's constant (J·s)
hbar = h / (2 * np.pi)  # Reduced Planck constant
Sc = 0.967
# Sc: scattering coefficient (~0.967), an empirical factor from fitting
# Coulomb scattering theory to experiments.

# ===============================
# Input parameters (need to be set)
# ===============================

V = 30000            # Accelerating voltage (V)
Lambda = 1.226e-9 / (V**0.5)
# Electron wavelength (m), from accelerating voltage V
Br = 1e8             # Reduced brightness (A/m^2·sr·V)
alpha = 5e-3         # Aperture semi-angle (rad)
d_source = 30e-9     # Source size (m)
M = 1000             # Magnification (from source to image plane)
dE = 0.5             # Energy spread (eV)

# Aberration coefficients
Cs = 1e-3            # Spherical aberration coefficient (m)
Cc = 1e-3            # Chromatic aberration coefficient (m)


"""
RPS of J. E. Barth and P. Kruit method for calculating probe size (FW50 value):
d_p = ( ( (((d_a^4 + d_s^4)^(1/4))^1.3 + d_geo^1.3 )^(1/1.3))^2 + d_c^2 + d_{e-e}^2 )^(1/2)



Components of probe size:

Geometric image of the source (d_geo) - only this contribution carries the current:
d_geo = M * d_v = (2/π) * sqrt(I_p / (B_r * V))* (1/alpha)

Diffraction contribution (d_a):
d_a = 0.54 * λ / alpha = 0.54 * (Λ / V^(1/2)) * (1/alpha)
where Λ = 1.226e-9 m·V^(1/2)

Spherical aberration contribution (d_s):
d_s = 0.18 * C_s * alpha^3

Chromatic aberration contribution (d_c):
d_c = 0.6 * C_c * (ΔU/U) * alpha

Current in the probe:
I_p = B_r * (π/4) * (d_geo)^2 * π * alpha^2 * V
"""