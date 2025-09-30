import traceback
from typing import Any, Optional, Tuple, List, Dict
import numpy as np
import scipy.sparse as sp  # Is this needed
import scipy.sparse.linalg as spla  # Is this needed
from scipy.linalg import expm


# Helpers
def secondary_kraus(K: np.ndarray, d: int) -> List[np.ndarray]:
    """
    Python translation of the MATLAB function secondary_kraus(K, d).

    K : numpy array, shape (d, d)
    d : integer (dimension)

    Returns
    -------
    M_list : list of numpy arrays
    Mdagger_list : list of numpy arrays
    """
    M_list = []

    for mm in range(d):
        temp = np.zeros((d, d), dtype=complex)

        for i in range(d):
            for j in range(d):
                # Create standard basis vectors
                m = np.zeros(d, dtype=complex)
                u = np.zeros(d, dtype=complex)
                v = np.zeros(d, dtype=complex)
                o = np.zeros(d, dtype=complex)

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
        List of d x d Kraus operators
    info : dict
        Contains Phi, Choi, eigvals, pos_idx, nK, trace_error, d, runtime
    Populations : np.ndarray
        Populations [s_population, c_population, s_population_K, c_population_K]
    Trace_error : np.ndarray
        Trace error values across time steps
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
            "Ns": 7,  # Fock cutoff for mode a
            "Nc": 7,  # Fock cutoff for mode b
            "hbar": 1.0,
            "zi": 1.0,
            "kappa_a": 0.02,
            "kappa_b": 0.02,
            # "t": 2.0 * np.pi,
            "tol": 1e-12,
            "coh": 0,
            "signal_photon": 1,
            "convsn_photon": 0,
            "time": np.linspace(1e-6, 50, 10),
        }

    Ns = params["Ns"]
    Nc = params["Nc"]
    hbar = params["hbar"]
    zi = params["zi"]
    kappa_a = params["kappa_a"]
    kappa_b = params["kappa_b"]
    # tprop = params["t"]
    tol = params["tol"]

    coh = params["coh"]
    signal_photon = params["signal_photon"]
    convsn_photon = params["convsn_photon"]
    time_grid = params["time"]

    # time_grid = np.linspace(tprop/100, tprop, 100)

    # Dimensions
    # ds = Ns + 1
    # dc = Nc + 1
    ds = Ns
    dc = Nc
    d = ds * dc

    # Initial state
    if initial_state is None:
        if coh == 1:
            alpha = signal_photon
            n = np.arange(ds)
            # I would use scipy.special.factorial instead (future proof)
            coh_s = np.exp(-0.5 * np.abs(alpha) ** 2) * (
                alpha**n / np.sqrt(np.math.factorial(n))
            )
            Fock_c = np.zeros(dc)
            Fock_c[convsn_photon] = 1
            psi0 = np.kron(coh_s, Fock_c)
            rho0 = np.outer(psi0, psi0.conj())
        else:
            Fock_s = np.zeros(ds)
            Fock_s[signal_photon] = 1
            Fock_c = np.zeros(dc)
            Fock_c[convsn_photon] = 1
            psi0 = np.kron(Fock_s, Fock_c)
            rho0 = np.outer(psi0, psi0.conj())
    else:
        assert isinstance(initial_state, np.ndarray)
        rho0 = initial_state

    # Annihilation operator
    def ann(N):
        return np.diag(np.sqrt(np.arange(1, N)), 1)

    a_s_single = ann(Ns)
    a_c_single = ann(Nc)

    Ia = np.eye(ds)
    Ib = np.eye(dc)

    a_s = np.kron(a_s_single, Ib)
    a_c = np.kron(Ia, a_c_single)

    a_s_d = a_s.conj().T
    a_c_d = a_c.conj().T

    # Hamiltonian
    H = 1j * hbar * (np.conj(zi) * a_c_d @ a_s - zi * a_s_d @ a_c)

    # Lindblad jump operators
    L1 = np.sqrt(kappa_a) * a_s
    L2 = np.sqrt(kappa_b) * a_c
    Lj = [L1, L2]

    # Liouvillian
    Id = np.eye(d)
    LH = -1j / hbar * (np.kron(H, Id) - np.kron(Id, H.T))

    Ldis = np.zeros((d**2, d**2), dtype=complex)
    for L in Lj:
        LdagL = L.conj().T @ L
        term1 = np.kron(L, L.conj())
        term2 = -0.5 * np.kron(LdagL, Id)
        term3 = -0.5 * np.kron(Id, LdagL.T)
        Ldis += term1 + term2 + term3

    L_super = LH + Ldis

    Populations = []
    Trace_rho = []
    Trace_error = []
    Kraus = []

    # Bind the values, these are overwritten in every step
    Phi = 0
    C = 0
    eigvals = None
    pos_idx = np.array([])
    trace_error = 0

    for t in time_grid:
        Phi = expm(L_super * t)
        rho_t = Phi @ rho0.flatten()
        rho_t = rho_t.reshape((d, d))

        trace_rho_L = np.trace(rho_t)
        s_population = np.trace(rho_t @ (a_s_d @ a_s))
        c_population = np.trace(rho_t @ (a_c_d @ a_c))

        # Choi matrix
        Phi4 = Phi.reshape(d, d, d, d)
        Phi_perm = np.transpose(Phi4, (0, 2, 1, 3))
        C = Phi_perm.reshape(d**2, d**2)
        C = (C + C.conj().T) / 2

        eigvals, VV = np.linalg.eigh(C)
        pos_idx = np.where(np.real(eigvals) > tol)[0]
        eigvals_pos = np.real(eigvals[pos_idx])
        Vpos = VV[:, pos_idx]

        Kraus = []
        for k in range(len(eigvals_pos)):
            vecGamma = Vpos[:, k]
            matGamma = vecGamma.reshape((d, d))
            Kraus.append(np.sqrt(eigvals_pos[k]) * matGamma)

        # Trace preservation check
        S = np.zeros((d, d), dtype=complex)
        rho_t_Kraus = np.zeros((d, d), dtype=complex)
        for K in Kraus:
            S += K.conj().T @ K
            rho_t_Kraus += K @ rho0 @ K.conj().T

        trace_error = np.linalg.norm(S - np.eye(d), "fro")
        Trace_error.append(trace_error)
        Trace_rho.append([trace_rho_L, np.trace(rho_t_Kraus)])

        s_population_K = np.trace(rho_t_Kraus @ (a_s_d @ a_s))
        c_population_K = np.trace(rho_t_Kraus @ (a_c_d @ a_c))
        Populations.append([s_population, c_population, s_population_K, c_population_K])

    runtime = time.time() - t0

    info = {
        "Phi": Phi,
        "Choi": C,
        "eigvals": eigvals,
        "trace_error": trace_error,
        "dimensions": d,
        "runtime": runtime,
        "operator_num_M": len(Kraus),
        "operator_num_K": len(Kraus),
    }

    M = []
    try:
        for K in Kraus:
            M.extend(secondary_kraus(K, Nc))
        info["operator_num_M"] = len(M)
    except Exception as e:
        print(f"K.shape={K.shape}", flush=True)
        print(f"Nc={Nc}", flush=True)
        traceback.print_exc()

    return (
        Kraus,
        M,
        info,
        np.array(Populations),
        np.array(Trace_error),
        np.array(Trace_rho),
    )
