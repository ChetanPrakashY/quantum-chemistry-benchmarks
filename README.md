# quantum-chemistry-benchmarks

From molecule to qubits and back: classical quantum-chemistry methods against exact references, a UCCSD-VQE built
from scratch (Pauli algebra, Jordan-Wigner, exact-gradient optimisation), the measurement cost of running it on
hardware, and a machine-learning correction from Hartree-Fock to CCSD(T).

**Results and discussion: [REPORT.md](REPORT.md).**

| File | What it does |
|---|---|
| `qcb/pauli.py` | Pauli strings in symplectic form: products, sums, sparse matrices; Jordan-Wigner fermion operators |
| `qcb/hamiltonian.py` | PySCF active space -> qubit Hamiltonian; Hartree-Fock state; number and S^2 operators |
| `qcb/vqe.py` | UCCSD generators, CNOT estimate, VQE with an exact forward/backward gradient, multiple layers, spin penalty |
| `benchmark.py` | H2, LiH, BeH2, H2O at three bond lengths: HF, MP2, CCSD, CCSD(T) vs FCI; VQE (1 and 2 layers) vs CASCI |
| `measurement.py` | shots needed for chemical accuracy: worst-case bound, terms separately, qubit-wise commuting groups |
| `delta_ml.py` | kernel ridge regression on Coulomb matrices: CCSD(T) directly vs the CCSD(T) - HF correction |
| `tests/` | the qubit Hamiltonian reproduces CASCI and HF to 1e-8 Ha; the analytic gradient matches finite differences; VQE is exact for LiH(2e,3o) |

Highlights:

- CCSD(T) is within chemical accuracy at equilibrium but misses stretched water by about 12 mHa, below the exact energy.
- At stretched geometries a triplet can lie below the singlet; without a spin constraint VQE slides into it. With an
  S^2 penalty, two UCCSD layers reach the exact singlet everywhere in this benchmark; one layer misses BeH2 by 7.5 mHa.
- Grouping commuting Pauli terms cuts the shots for chemical accuracy several-fold (LiH, water).

```bash
pip install pyscf        # plus numpy, scipy, scikit-learn, matplotlib
./run_all.sh
```
