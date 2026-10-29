"""UCCSD ansatz and VQE on an exact statevector. The state is a product of exponentials
U = exp(theta_K G_K) ... exp(theta_1 G_1) acting on Hartree-Fock, with G_k = T_k - T_k^dagger (anti-Hermitian), and
the energy gradient is computed exactly with one forward and one backward sweep (no finite differences)."""
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from .pauli import jw_annihilation, dagger, op_mul, op_add, clean, to_sparse, weight


def uccsd_generators(n_qubits, n_electrons):
    """Spin-conserving singles and doubles from occupied to virtual spin orbitals, as Pauli dicts."""
    a = [jw_annihilation(j) for j in range(n_qubits)]
    ad = [dagger(x) for x in a]
    occ, vir = range(n_electrons), range(n_electrons, n_qubits)
    gens, labels = [], []
    for i, b in itertools.product(occ, vir):
        if i % 2 == b % 2:
            T = op_mul(ad[b], a[i])
            gens.append(clean(op_add(T, dagger(T), -1.0))); labels.append(f"{i}->{b}")
    for (i, j), (b, c) in itertools.product(itertools.combinations(occ, 2), itertools.combinations(vir, 2)):
        if (i % 2 + j % 2) == (b % 2 + c % 2):
            T = op_mul(op_mul(ad[c], ad[b]), op_mul(a[j], a[i]))
            gens.append(clean(op_add(T, dagger(T), -1.0))); labels.append(f"{i},{j}->{b},{c}")
    return gens, labels


def cnot_estimate(gens):
    """CNOTs to exponentiate every Pauli string of every generator with the standard parity ladder, 2(w - 1) each."""
    return sum(2 * (weight(p) - 1) for g in gens for p in g if weight(p) > 0)


def objective(H_sparse, gens_sparse, psi0):
    """Returns f(theta) -> (energy, exact gradient) and the state builder."""
    def states(theta):
        phis = [psi0]
        for t, G in zip(theta, gens_sparse):
            phis.append(expm_multiply(t * G, phis[-1]))
        return phis

    def energy_and_grad(theta):
        phis = states(theta)
        psi = phis[-1]
        hpsi = H_sparse @ psi
        e = np.vdot(psi, hpsi).real
        lam, grad = hpsi, np.zeros(len(theta))
        for k in range(len(theta) - 1, -1, -1):           # backward sweep: lam = U_N^+..U_{k+1}^+ H psi
            grad[k] = 2 * np.real(np.vdot(lam, gens_sparse[k] @ phis[k + 1]))
            lam = expm_multiply(-theta[k] * gens_sparse[k], lam)
        return e, grad

    return energy_and_grad, states


def run_vqe(H_sparse, gens_sparse, psi0, layers=1, penalty=None, starts=1, seed=0, maxiter=5000):
    """Minimise <H> (+ <penalty> if given, e.g. mu S^2 to stay on singlets) with `layers` repetitions of the
    generator list. starts > 1: small random starting points, best kept (start 0 is near zero)."""
    G = list(gens_sparse) * layers
    f, states = objective(H_sparse + penalty if penalty is not None else H_sparse, G, psi0)
    rng = np.random.default_rng(seed)
    best = None
    for k in range(starts):
        x0 = rng.normal(0, 0.05, len(G)) if k else np.full(len(G), 1e-3)
        r = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-10, ftol=1e-15))
        if best is None or r.fun < best.fun:
            best = r
    psi = states(best.x)[-1]
    return np.vdot(psi, H_sparse @ psi).real, best.x, psi, best.nit
