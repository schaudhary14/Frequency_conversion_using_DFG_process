import numpy as np
import matplotlib.pyplot as plt

from channel_model import Kraus_DFG

# Parameters for Kraus operators
params = {
    "Ns": 5,  # Fock cutoff for mode a (pump)
    "Nc": 5,  # Fock cutoff for mode b (SH)
    "hbar": 1.0,  # set to 1 for convenience
    "zi": 1.0,  # chi(2) coupling proportional to pump amplitude
    "kappa_a": 0.05,  # loss rate a
    "kappa_b": 0.01,  # loss rate b
    # "t": 5 * np.pi,     # propagation time
    "tol": 1e-12,  # numeric tolerance
    "coh": 0,  # signal mode
    "signal_photon": 1,
    "convsn_photon": 0,
    "time": np.linspace(1e-6, 50, 10),
}

# Call the main function
Kraus, info, Populations, Trace_error, Trace_rho = Kraus_DFG(params)
