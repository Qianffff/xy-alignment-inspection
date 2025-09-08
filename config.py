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
Lambda = 1.226e-9

# ===============================
# Input parameters (need to be set)
# ===============================

V = 300000           # Accelerating voltage (V)
elambda = Lambda / (V**0.5)
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
    """
    Calculate the effective electron probe size.

    Parameters:
    I_p: probe current (A)

    Returns: 
    d_p: effective probe size (m)
    """

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



def find_optimal_probe_current():
    """
    Find the optimal probe current that minimizes the probe size and plot the results.
    
    Returns:
    I_optimal: optimal probe current (A)
    d_min: minimum probe size (m)
    """
    # Generate current values
    I_values = np.logspace(-14, -9, 200)
    d_values = np.array([d_p_func(I) for I in I_values])
    
    # Find the minimum probe size and corresponding current
    min_index = np.argmin(d_values)
    I_optimal = I_values[min_index]
    d_min = d_values[min_index]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot( d_values*1e9, I_values*1e12, label='Probe Size vs Current')  # Convert to pA and nm
    plt.scatter( d_min*1e9, I_optimal*1e12, color='red', s=50, 
                label=f'Optimal: {I_optimal*1e12:.2f} pA, {d_min*1e9:.2f} nm')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Probe Current I_p (pA)")
    plt.xlabel("Probe Diameter d_p (nm)")
    plt.title("Probe Size vs. Probe Current")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.show()
    
    return I_optimal, d_min

# Find and display the optimal probe current
I_optimal, d_min = find_optimal_probe_current()
print(f"Optimal probe current: {I_optimal:.2e} A")
print(f"Minimum probe size: {d_min:.2e} m ({d_min*1e9:.2f} nm)")