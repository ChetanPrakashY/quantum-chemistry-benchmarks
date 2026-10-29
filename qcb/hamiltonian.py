"""Molecule -> active-space integrals (PySCF) -> qubit Hamiltonian (Jordan-Wigner), as a dict of Pauli strings.
Spin orbitals are interleaved: qubit 2p is spatial orbital p spin up, qubit 2p+1 spin down."""
import itertools
import numpy as np
from pyscf import gto, scf, mcscf
from .pauli import jw_annihilation, dagger, op_mul, op_add, clean


def active_space(atom, basis, ncas, nelecas, charge=0, spin=0):
    """RHF, then the CASCI active space: (core energy, one-body h, two-body (pq|rs)) and the exact CASCI energy."""
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fix_spin_(ss=spin / 2 * (spin / 2 + 1))   # target the right spin state: for stretched bonds a triplet can lie lower
    mc.verbose = 0
    e_casci = mc.kernel()[0]
    h1, e_core = mc.get_h1eff()
    h2 = mcscf.casci.ao2mo.restore(1, mc.get_h2eff(), ncas)
    return dict(e_core=e_core, h1=h1, h2=h2, e_hf=mf.e_tot, e_casci=e_casci, ncas=ncas, nelecas=nelecas, mol=mol, mf=mf)


def qubit_hamiltonian(e_core, h1, h2, tol=1e-10):
    """H = E_core + sum h_pq a+_p a_q + 1/2 sum (pq|rs) a+_ps a+_rt a_rt' ... (spin-summed, chemists' notation)."""
    n = h1.shape[0]
    a = [jw_annihilation(j) for j in range(2 * n)]
    ad = [dagger(x) for x in a]
    H = {(0, 0): e_core}
    for p, q in itertools.product(range(n), repeat=2):
        if abs(h1[p, q]) < tol:
            continue
        for s in (0, 1):
            H = op_add(H, op_mul(ad[2 * p + s], a[2 * q + s]), h1[p, q])
    pairs = {}
    for s, t in itertools.product((0, 1), repeat=2):
        for p, q, r, u in itertools.product(range(n), repeat=4):
            g = h2[p, q, r, u]
            if abs(g) < tol or (2 * p + s == 2 * r + t) or (2 * q + s == 2 * u + t):
                continue
            key = (2 * p + s, 2 * r + t, 2 * u + t, 2 * q + s)       # a+_ps a+_rt a_ut a_qs
            pairs[key] = pairs.get(key, 0) + 0.5 * g
    for (i, j, k, l), g in pairs.items():
        H = op_add(H, op_mul(op_mul(ad[i], ad[j]), op_mul(a[k], a[l])), g)
    return {p: c.real for p, c in clean(H).items()}             # H is Hermitian: imaginary parts cancel


def hf_state(n_qubits, n_electrons):
    """Hartree-Fock determinant: the lowest n_electrons spin orbitals filled."""
    v = np.zeros(2 ** n_qubits, complex)
    v[(1 << n_electrons) - 1] = 1.0
    return v


def number_operator(n_qubits):
    a = [jw_annihilation(j) for j in range(n_qubits)]
    N = {}
    for j in range(n_qubits):
        N = op_add(N, op_mul(dagger(a[j]), a[j]))
    return clean(N)


def s_squared(n_qubits):
    """Total spin S^2 = S- S+ + Sz + Sz^2 as a Pauli dict (interleaved spin orbitals)."""
    a = [jw_annihilation(j) for j in range(n_qubits)]
    ad = [dagger(x) for x in a]
    Sp, Sz = {}, {}
    for p in range(n_qubits // 2):
        Sp = op_add(Sp, op_mul(ad[2 * p], a[2 * p + 1]))
        Sz = op_add(Sz, op_mul(ad[2 * p], a[2 * p]), 0.5)
        Sz = op_add(Sz, op_mul(ad[2 * p + 1], a[2 * p + 1]), -0.5)
    S2 = op_add(op_mul(dagger(Sp), Sp), Sz)
    return clean(op_add(S2, op_mul(Sz, Sz)))
