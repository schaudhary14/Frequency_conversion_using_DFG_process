import numpy as np
import matplotlib.pyplot as plt

from channel_model import Kraus_DFG

# Parameters for Kraus operators
params = {
    "Ns": 4,              # Fock cutoff for mode "signal"
    "Nc": 4,              # Fock cutoff for mode "converted"
    "hbar": 1.0,          #  physical constant
    "zi": 1.0,            # pump amplitude
    "gamma_s": 0.05,      # damping rate of signal mode
    "gamma_c": 0.01,      # damping rate of converted mode
    "g_sc": 0.8,          # signal to converted mode transfer rate
    "g_Raman_s": 1e-5,    # Raman coupling to signal mode
    "g_Raman_c": 1e-4,    # Raman coupling to converted mode
    "tol": 1e-12,         # tolerence value to numerically calculate eigenvalues
    "coh": 0,             # coh == 1 means input signal state is coherent state with mean photon number 'signal_photon', else coh == 1 means Fock state
    "signal_photon": 1,   # photon number in signal mode at t=0
    "convsn_photon": 0,   # photon number in conversion mode at t=0
    "time": np.linspace(1e-6, 50, 10), # evolve system for given time instants
}

# Call the main function
Kraus, M, info, Fidelity, Populations, Trace_error, Trace_rho = Kraus_DFG(params)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Top-left
axes[0].plot(params["time"], Populations[:,2], ".-", label="signal mode")
axes[0].plot(params["time"], Populations[:,3], ".-", label="converted mode")
axes[0].set_title("Populations")
axes[0].set_xlabel("time [s]")
axes[0].set_ylabel("<n>")
axes[0].legend()

# Top-right
axes[1].plot(params["time"], Fidelity, ".-")
axes[1].set_title("Fidelity")
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("Fidelity")
axes[1].legend()

plt.tight_layout()
plt.show()
