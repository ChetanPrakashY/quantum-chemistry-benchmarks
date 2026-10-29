"""python -m unittest discover -s tests -v"""
import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qcb.pauli import to_sparse  # noqa: E402
from qcb.hamiltonian import active_space, qubit_hamiltonian, hf_state, number_operator  # noqa: E402
from qcb.vqe import uccsd_generators, run_vqe, objective  # noqa: E402


class Chemistry(unittest.TestCase):
    def test_qubit_hamiltonian_reproduces_casci_and_hf(self):
        for atom, ncas, ne in [("H 0 0 0; H 0 0 0.74", 2, 2), ("Li 0 0 0; H 0 0 1.6", 3, 2)]:
            a = active_space(atom, "sto-3g", ncas, ne)
            H = to_sparse(qubit_hamiltonian(a["e_core"], a["h1"], a["h2"]), 2 * ncas)
            N = to_sparse(number_operator(2 * ncas), 2 * ncas).diagonal().real
            sector = np.where(np.isclose(N, ne))[0]
            e0 = np.linalg.eigvalsh(H.toarray()[np.ix_(sector, sector)])[0]
            self.assertAlmostEqual(e0, a["e_casci"], places=8)
            hf = hf_state(2 * ncas, ne)
            self.assertAlmostEqual(np.vdot(hf, H @ hf).real, a["e_hf"], places=8)

    def test_gradient_matches_finite_differences(self):
        a = active_space("Li 0 0 0; H 0 0 1.6", "sto-3g", 3, 2)
        H = to_sparse(qubit_hamiltonian(a["e_core"], a["h1"], a["h2"]), 6)
        gens, _ = uccsd_generators(6, 2)
        G = [to_sparse(g, 6) for g in gens]
        f, _ = objective(H, G, hf_state(6, 2))
        t0 = np.random.default_rng(0).normal(0, 0.1, len(G)); h = 1e-6
        fd = np.array([(f(t0 + h * e)[0] - f(t0 - h * e)[0]) / (2 * h) for e in np.eye(len(G))])
        self.assertLess(np.abs(f(t0)[1] - fd).max(), 1e-6)

    def test_vqe_reaches_casci_for_lih(self):
        a = active_space("Li 0 0 0; H 0 0 1.6", "sto-3g", 3, 2)
        H = to_sparse(qubit_hamiltonian(a["e_core"], a["h1"], a["h2"]), 6)
        gens, _ = uccsd_generators(6, 2)
        e, *_ = run_vqe(H, [to_sparse(g, 6) for g in gens], hf_state(6, 2))
        self.assertLess(abs(e - a["e_casci"]), 1e-6)       # two electrons: UCCSD is exact


if __name__ == "__main__":
    unittest.main()
