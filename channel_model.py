import traceback
from typing import Any, Optional, Tuple, List, Dict
import numpy as np
import scipy.sparse as sp  # Is this needed
import scipy.sparse.linalg as spla  # Is this needed
from scipy.linalg import expm


# Helpers
def secondary_kraus(K: np.ndarray, ds: int) -> List[np.ndarray]:
    """
    Find the Kraus operators "M" (ds x ds) from the Kraus operators "K" (d x d)
    ρ_sc(t) = Σ_i K_i ρ_sc(0) K†_i
    ρ_c(t) = Σ_i M_i ρ_s(0) M†_i

    K : numpy array, shape (d, d)
    d : integer (dimension)

    Returns
    -------
    M_list : list of numpy arrays each (ds x ds)
    """
    M_list = []

    for mm in range(ds):
        temp = np.zeros((ds, ds), dtype=complex)

        for i in range(ds):
            for j in range(ds):
                # Create standard basis vectors
                m = np.zeros(ds, dtype=complex)
                u = np.zeros(ds, dtype=complex)
                v = np.zeros(ds, dtype=complex)
                o = np.zeros(ds, dtype=complex)

                m[mm] = 1
                u[i] = 1
                v[j] = 1
                o[0] = 1

                bra = np.kron(m, u).conjugate().T  # row vector
                ket = np.kron(v, o)  # column vector

                temp += bra @ K @ ket * np.outer(u, v)

        M_list.append(temp)
    return M_list


def Kraus_DFG(params=None, initial_state: Optional[np.ndarray] = None) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute finite-time Kraus operators for chi(2) DFG model.

    Parameters
    ----------
    params : dict
        Dictionary of parameters. If None, defaults are used.
    initial_state: np.ndarray, optional
        Optional initial state, first mode must be signal
        second one output mode

    Returns
    -------
    Kraus : list of np.ndarray
        List of (d x d) Kraus operators
    info : dict
        Contains Phi, Choi, eigvals, norm_error, dimensions_K, dimensions_M, runtime, operator_num_M, operator_num_K
    Populations : np.ndarray
        Populations [s_population, c_population, s_population_K, c_population_K]
    Norm2_error : np.ndarray
        Norm2 error values across time steps
    Trace_rho : np.ndarray
        Trace of rho(t) vs. reconstructed via Kraus

    Notes
    -----
    - If `initial_state` is not given a synthetic initial state is created
    - `initial_state` is given the initial corresponding dimensions in
      the params `Ns`, `Nc` should reflect the dimensions of the given state
    """

    import time

    t0 = time.time()

    # Default parameters
    if params is None or len(params) == 0:
        params = {
            "Ns": 7,          # Fock cutoff for mode "signal"
            "Nc": 7,          # Fock cutoff for mode "converted"
            "hbar": 1.0,      # physical constant
            "zi": 1.0,        # pump amplitude
            "gamma_s": 0.02,  # damping rate of signal mode
            "gamma_c": 0.02,  # damping rate of converted mode
            "g_sc": 0.8,      # signal to converted mode transfer rate
            "g_Raman_s": 1e-4,# Raman coupling to signal mode
            "g_Raman_c":1e-3, # Raman coupling to converted mode
            "tol": 1e-12,     # tolerence value to numerically calculate eigenvalues
            "coh": 0,         # coh == 1 means input signal state is coherent state with mean photon number 'signal_photon', else coh == 1 means Fock state
            "signal_photon": 1, # photon number in signal mode at t=0
            "convsn_photon": 0, # photon number in conversion mode at t=0
            "time": np.linspace(1e-6, 50, 10), # evolve system for given time instants
            
        }

    Ns = params["Ns"]
    Nc = params["Nc"]
    hbar = params["hbar"]
    zi = params["zi"]
    gamma_s = params["gamma_s"]
    gamma_c = params["gamma_c"]
    g_sc = params["g_sc"]
    g_Raman_s = params["g_Raman_s"]
    g_Raman_c = params["g_Raman_c"]
    tol = params["tol"]
    coh = params["coh"]
    signal_photon = params["signal_photon"]
    convsn_photon = params["convsn_photon"]
    time_grid = params["time"]

    # Dimensions
    ds = Ns            # dimension of signal mode; Foack basis: {|0>, |1>, .. |ds-1>}
    dc = Nc            # dimension of converted mode; Foack basis: {|0>, |1>, .. |dc-1>}
    d = ds * dc        # dimension of bipartite Hilbert space (d X d)

    # Generating the Initial state of the signal and the converted mode
    if initial_state is None:
        if coh == 1:               # signal state is coherent state at t=0
            alpha = signal_photon
            n = np.arange(ds)
            coh_s = np.exp(-0.5 * np.abs(alpha) ** 2) * (
                alpha**n / np.sqrt(np.math.factorial(n))   # I would use scipy.special.factorial instead (future proof)
            )
            Fock_c = np.zeros(dc)
            Fock_c[convsn_photon] = 1
            psi0 = np.kron(coh_s, Fock_c)            # (|ψ(0)> = |signal mode> |converted mode>)
            rho0 = np.outer(psi0, psi0.conj())       # density matrix of the combined system at t=0 (|signal mode> |converted mode>)
        else:                      # signal state is Fock state at t=0
            Fock_s = np.zeros(ds)
            Fock_s[signal_photon] = 1
            Fock_c = np.zeros(dc)
            Fock_c[convsn_photon] = 1
            psi0 = np.kron(Fock_s, Fock_c)             # (|ψ(0)> = |signal mode> |converted mode>)
            rho0 = np.outer(psi0, psi0.conj())         # density matrix of the combined system at t=0 
    else:
        assert isinstance(initial_state, np.ndarray)
        rho0 = initial_state

    #Initial state of signal
    rho0_s = rho0.reshape(ds, dc, ds, dc)
    rho0_s = np.trace(rho0_s, axis1=1, axis2=3)
    
    # Annihilation operator
    def ann(N):
        return np.diag(np.sqrt(np.arange(1, N)), 1)

    a_s_single = ann(ds) # annihilation operator for signal mode (ds x ds)
    a_c_single = ann(dc) # annihilation operator for converted mode (dc x dc)

    Ia = np.eye(ds) # Identity matrix (ds x ds)
    Ib = np.eye(dc) # Identity matrix (dc x dc)

    a_s = np.kron(a_s_single, Ib) # a_s ⊗ I_c (d x d)
    a_c = np.kron(Ia, a_c_single) # I_s ⊗ a_c (d x d)

    a_s_d = a_s.conj().T # Creation operator: a†_s ⊗ I_c (d x d)
    a_c_d = a_c.conj().T # Creation operator: I_s ⊗ a†_c (d x d)

    # Hamiltonian for the DFG process
    H = 1j * hbar * (np.conj(zi) * a_c_d @ a_s - zi * a_s_d @ a_c) # iħ(ξ* a†_s ⊗ a_c - ξ a_s ⊗ a†_c)

    # Lindblad jump operators
    L1 = np.sqrt(gamma_s) * a_s      # damping of signal mode, dim: (d x d)
    L2 = np.sqrt(gamma_c) * a_c      # damping of converted mode, dim: (d x d)
    L3 = np.sqrt(g_sc) * a_c_d @ a_s # transfer of photon from signal mode to converted mode, dim: (d x d)
    L4 = np.sqrt(g_Raman_s) * a_s_d  # Raman coupling to signal mode, dim: (d x d)
    L5 = np.sqrt(g_Raman_c) * a_c_d  # Raman coupling to converted mode, dim: (d x d)
    Lj = [L1, L2, L3, L4, L5]
    
    # Liouvillian
    Id = np.eye(d)
    LH = -1j / hbar * (np.kron(H, Id) - np.kron(Id, H.T))

    Ldis = np.zeros((d**2, d**2), dtype=complex) # calculating dissipators for all the above Lindblad operators
    for L in Lj:
        LdagL = L.conj().T @ L
        term1 = np.kron(L, L.conj())
        term2 = -0.5 * np.kron(LdagL, Id)
        term3 = -0.5 * np.kron(Id, LdagL.T)
        Ldis += term1 + term2 + term3

    L_super = LH + Ldis # Liouvillian super operator; (d^2 x d^2)

    Populations = [] # <a†a> for signal and converted mode computed from both methods: 1. Master equation 2. Kraus operators
    Trace_rho = []   # Trace[ρ_sc(t)] computed from both methods: 1. Master equation 2. Kraus operators
    Norm2_error = [] # || ρ_sc(t) - I ||
    Fidelity = []    # | Trace[M ρ_s(0)] |^2, metric for error calculation

    # Initializing values for:
    Phi = 0                 # exponent of superoperator
    C = 0                   # Choi matrix
    eigvals = None          # eigenvalues of Choi matrix
    pos_idx = np.array([])  # index of eigenvalues above tolerence value
    norm_error = 0         # Trace error

    for t in time_grid:
        Phi = expm(L_super * t)       # exponent of superoperator; (d^2 x d^2)
        rho_t = Phi @ rho0.flatten()  # |ρ_sc(t)> ; (d^2 x 1)
        rho_t = rho_t.reshape((d, d)) # ρ_sc(t) ; (d x d)

        trace_rho_L = np.trace(rho_t)                  # Trace[ρ_sc(t)], calculated from Liouvillian method
        s_population = np.trace(rho_t @ (a_s_d @ a_s)) # signal mode population Trace[ρ_sc(t) a†_s a_s], calculated from Liouvillian method
        c_population = np.trace(rho_t @ (a_c_d @ a_c)) # converted mode population Trace[ρ_sc(t) a†_c a_c], calculated from Liouvillian method

        # Choi matrix
        Phi4 = Phi.reshape(d, d, d, d)                # reshaping exponent of superoperator from (d^2 x d^2) ----> (d x d x d x d)
        Phi_perm = np.transpose(Phi4, (0, 2, 1, 3))   # indexing rearranging
        C = Phi_perm.reshape(d**2, d**2)              # reshaping to (d^2 x d^2)
        C = (C + C.conj().T) / 2                      # ensuring hermiticity

        eigvals, VV = np.linalg.eigh(C)               # Eigenvalues and eigenvectors of Choi matrix
        pos_idx = np.where(np.real(eigvals) > tol)[0] # indexing eigenvalues above tolerence
        eigvals_pos = np.real(eigvals[pos_idx])       # Eigenvalues (keeping real part only)
        Vpos = VV[:, pos_idx]                         # corresponding eigenvectors

        Kraus = []
        for k in range(len(eigvals_pos)):
            vecGamma = Vpos[:, k]
            matGamma = vecGamma.reshape((d, d))
            Kraus.append(np.sqrt(eigvals_pos[k]) * matGamma)  # List of Kraus operators K; (d x d)

        # Trace preservation check
        S = np.zeros((d, d), dtype=complex)
        rho_t_Kraus = np.zeros((d, d), dtype=complex)
        for K in Kraus:
            S += K.conj().T @ K                                # Σ_i K†_i K_i
            rho_t_Kraus += K @ rho0 @ K.conj().T               # ρ_sc(t) = Σ_i K_i ρ_sc(0) K†_i, calculated using Kraus formalism

        norm_error = np.linalg.norm(S - np.eye(d), "fro")      #  || Σ_i K†_i K_i - I ||, Frobenius Norm distance
        Norm2_error.append(norm_error)
        Trace_rho.append([trace_rho_L, np.trace(rho_t_Kraus)]) # Trace[ ρ_sc(t) ], computed from both methods: 1. Master equation 2. Kraus operators

        s_population_K = np.trace(rho_t_Kraus @ (a_s_d @ a_s)) # signal mode population Trace[ρ_sc(t) a†_s a_s], calculated from Kraus operator method
        c_population_K = np.trace(rho_t_Kraus @ (a_c_d @ a_c)) # converted mode population Trace[ρ_sc(t) a†_s a_s], calculated from Kraus operator method
        Populations.append([s_population, c_population, s_population_K, c_population_K])

        M = []   # List of Kraus operators mapping ρ_s(0) ---> ρ_c(t), (ds x ds)
        try:
            for K in Kraus:
                M.extend(secondary_kraus(K, ds))  # Calculating M shape: (ds x ds) from K shape: (d x d)
            info["operator_num_M"] = len(M)
        except Exception as e:
            print(f"K.shape={K.shape}", flush=True)
            print(f"Nc={Nc}", flush=True)
            traceback.print_exc()
    
        fidelity = 0                             # Fidelity calculation
        for idx1 in range(len(M)):
            temp = M[idx1] @ rho0_s
            fidelity += abs(np.trace(temp)) ** 2  # | Trace[M ρ_s(0)] |^2, metric for error calculation

        Fidelity.append([fidelity])

    runtime = time.time() - t0

    info = {
        "Phi": Phi,
        "Choi": C,
        "eigvals": eigvals,
        "norm_error": norm_error,
        "dimensions_K": d,
        "dimensions_M": ds,
        "runtime": runtime,
        "operator_num_M": len(M),
        "operator_num_K": len(Kraus),
    }

     
    return (
        Kraus,                # list of Kraus operators K (d x d), at final time
        M,                    # list of Kraus operators M (Ns x Ns), at final time
        info,                 # relevant information at final time
        Fidelity,             # list of fidelity values for different times
        np.array(Populations),# 2D array of populations for different times [signal_L, converted_L, signal_K, converted_K]
        np.array(Norm2_error),# Norm error calculated for differnt times
        np.array(Trace_rho),  # Trace[ρ_sc(t)] calculated for different times [Liouvillian_method, Kraus_method]
    )
