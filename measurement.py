"""How many measurements (shots) a quantum computer needs to estimate the energy to chemical accuracy.

The energy is a sum of Pauli terms, E = sum_j c_j <P_j>. Terms measured together in one circuit form a group G_g; if a
group gets m_g shots, the variance of the total is sum_g Var(G_g)/m_g. Minimising the total shots M = sum m_g for a
target standard error eps (Lagrange multiplier) gives m_g proportional to sqrt(Var(G_g)) and

    M = (sum_g sqrt(Var(G_g)))^2 / eps^2,   Var(G_g) = <G_g^2> - <G_g>^2 in the state being measured.

Three strategies: every term on its own; qubit-wise commuting (QWC) groups, measurable with single-qubit rotations;
and the worst case bound (sum_j |c_j|)^2 / eps^2 that needs no knowledge of the state. Variances use the exact singlet
ground state, which is where a converged VQE ends up."""
import json
import numpy as np
from qcb.pauli import to_sparse, label
from qcb.hamiltonian import active_space, qubit_hamiltonian, s_squared
from benchmark_geoms import geom, ACTIVE

EPS = 1.594e-3                      # chemical accuracy, Ha


def qwc(p1, p2, n):
    """Qubit-wise commuting: on every qubit the two letters are equal or one is the identity."""
    a, b = label(p1, n), label(p2, n)
    return all(x == y or x == "I" or y == "I" for x, y in zip(a, b))


def greedy_groups(H, n):
    groups = []
    for p in sorted((p for p in H if p != (0, 0)), key=lambda p: -abs(H[p])):
        for g in groups:
            if all(qwc(p, q, n) for q in g):
                g.append(p); break
        else:
            groups.append([p])
    return groups


def singlet_ground(Hs, S2s, n, ne):
    H, S2 = Hs.toarray(), S2s.toarray()
    b = np.arange(2 ** n)
    N = np.array([bin(x).count("1") for x in b])
    sz = np.array([bin(x & int("01" * (n // 2), 2)).count("1") - bin(x & int("10" * (n // 2), 2)).count("1") for x in b])
    sec = np.where((N == ne) & (sz == 0))[0]
    w, v = np.linalg.eigh(H[np.ix_(sec, sec)] + 0.5 * S2[np.ix_(sec, sec)])   # the penalty lifts triplets
    psi = np.zeros(2 ** n, complex); psi[sec] = v[:, 0]
    return psi


rows = []
for name in ACTIVE:
    ncas, ne = ACTIVE[name]
    n = 2 * ncas
    a = active_space(geom(name, 1.0), "sto-3g", ncas, ne)
    H = qubit_hamiltonian(a["e_core"], a["h1"], a["h2"])
    psi = singlet_ground(to_sparse(H, n), to_sparse(s_squared(n), n), n, ne)
    def var(terms):
        G = to_sparse({p: H[p] for p in terms}, n)
        g = G @ psi
        return max(np.vdot(g, g).real - abs(np.vdot(psi, g)) ** 2, 0.0)
    terms = [p for p in H if p != (0, 0)]
    groups = greedy_groups(H, n)
    m_single = sum(np.sqrt(var([p])) for p in terms) ** 2 / EPS ** 2
    m_qwc = sum(np.sqrt(var(g)) for g in groups) ** 2 / EPS ** 2
    m_bound = sum(abs(H[p]) for p in terms) ** 2 / EPS ** 2
    rows.append(dict(molecule=name, qubits=n, terms=len(terms), groups=len(groups), shots_bound=m_bound,
                     shots_single=m_single, shots_qwc=m_qwc))
    print(f"{name:5} {n:2d} qubits {len(terms):4d} terms -> {len(groups):3d} QWC groups | shots for 1 kcal/mol: "
          f"bound {m_bound:9.2e}, terms separately {m_single:9.2e}, QWC groups {m_qwc:9.2e}")
json.dump(dict(eps_ha=EPS, rows=rows), open("results/measurement.json", "w"), indent=1)
