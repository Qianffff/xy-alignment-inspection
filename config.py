import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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

V = 300000           # Accelerating voltage (V)
Lambda = 1.226e-9 / (V**0.5)
# Electron wavelength (m), from accelerating voltage V
Br = 1e8             # Reduced brightness (A/m^2·sr·V)
alpha = 5e-3         # Aperture semi-angle (rad)
d_source = 30e-9     # Source size (m)
M = 1000             # Magnification (from source to image plane)
dU = 0.5             # Energy spread (eV)

# Aberration coefficients
Cs = 1e-3            # Spherical aberration coefficient (m)
Cc = 2               # Chromatic aberration coefficient (m)



# ===============================
# IRPS of J. E. Barth and P. Kruit method for calculating probe size (FW50 value)
# ===============================

def d_p_func(I_p):
    # Geometric contribution (depends on I_p)
    d_geo = (2/np.pi) * np.sqrt(I_p / (Br * V)) / alpha

    # Diffraction contribution
    d_a = 0.54 * Lambda / alpha

    # Spherical aberration
    d_s = 0.18 * Cs * alpha**3

    # Chromatic aberration
    d_c = 0.6 * Cc * (dU / V) * alpha

    # RPS combined probe size
    d_p = np.sqrt( ((d_a**4 + d_s**4)**0.25**1.3 + d_geo**1.3)**(1/1.3)**2 + d_c**2 )
    return d_p


# ===============================
# Generate I_p values (log scale)
# ===============================

I_values = np.logspace(-14, -9, 200)
d_values = np.array([d_p_func(I) for I in I_values])

# ===============================
# Plot
# ===============================

plt.figure(figsize=(6,4))
plt.plot(I_values*1e12, d_values*1e9)  # Convert to pA and nm
plt.xscale('log')
plt.xlabel("Probe Current I_p (pA)")
plt.ylabel("Probe Diameter d_p (nm)")
plt.title("Probe Size vs. Probe Current")
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()