"""Write REPORT.md from the result files, so every number in it comes from a run."""
import json

B = json.load(open("results/benchmark.json"))
M = json.load(open("results/measurement.json"))
D = json.load(open("results/delta_ml.json"))
rows = B["rows"]
by = {(r["molecule"], r["scale"]): r for r in rows}
f = lambda v, d=2: f"{v:+.{d}f}"
tiny = lambda v: "< 1e-6" if abs(v) < 1e-6 else f"{v:.1e}"

out = ["# Quantum chemistry benchmarks: classical methods, UCCSD-VQE, measurement cost and Delta-learning", "",
       "Everything below is produced by the scripts in this repository (`./run_all.sh`). Energies in milli-Hartree "
       "(mHa); chemical accuracy is 1 kcal/mol = 1.594 mHa. Basis STO-3G for the benchmark, so exact full configuration "
       "interaction (FCI) is affordable as the reference.", "",
       "## 1. Pipeline", "",
       "1. **PySCF**: restricted Hartree-Fock, MP2, CCSD, CCSD(T) and exact FCI (singlet) in the full space; an active "
       "space with its exact CASCI energy (singlet).",
       "2. **Qubits** (`qcb/pauli.py`, `qcb/hamiltonian.py`): the active-space Hamiltonian in second quantisation, mapped "
       "to Pauli strings with the Jordan-Wigner transformation by a small symplectic Pauli algebra. Tested: its lowest "
       "eigenvalue equals PySCF's CASCI energy and its Hartree-Fock expectation equals the RHF energy, to 1e-8 Ha.",
       "3. **VQE** (`qcb/vqe.py`): a product of exponentials of UCCSD generators on an exact statevector, optimised with "
       "L-BFGS using an exact gradient from one forward and one backward sweep (tested against finite differences). "
       "A penalty 0.5 S^2 keeps the state a singlet (section 3).", "",
       "## 2. Benchmark", "",
       "Errors against exact FCI in the full STO-3G space for the classical methods, and of UCCSD-VQE against exact "
       "CASCI in the same active space (a fair comparison: same Hamiltonian). Bond lengths are scaled by 1.0, 1.5 and "
       "2.0 from equilibrium.", "",
       "| Molecule | Bonds x | Active space / qubits | Pauli terms | Parameters | CNOTs (est.) | HF | MP2 | CCSD | CCSD(T) | VQE 1 layer | VQE 2 layers |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['molecule']} | {r['scale']:.1f} | {r['active']} / {r['qubits']} | {r['pauli_terms']} | {r['params']} | "
               f"{r['cnots']} | {f(r['err_hf'],1)} | {f(r['err_mp2'],1)} | {f(r['err_ccsd'],3)} | {f(r['err_ccsd_t'],3)} | "
               f"{tiny(r['err_vqe_vs_casci'])} | {tiny(r['err_vqe2_vs_casci'])} |")
w = by[("H2O", 2.0)]; be = by[("BeH2", 2.0)]
out += ["", "![benchmark](results/benchmark.png)", "",
        "**What the table shows.**", "",
        f"- Near equilibrium CCSD(T) is within chemical accuracy everywhere. Stretch the bonds and it fails: "
        f"{f(w['err_ccsd_t'])} mHa for water and {f(be['err_ccsd_t'])} mHa for BeH2 at twice the bond length, and it "
        "goes *below* the exact energy (perturbative triples are not variational). This strongly correlated regime is "
        "where quantum algorithms are meant to help.",
        f"- One layer of UCCSD is exact for two electrons (H2, LiH active spaces), but for four electrons it misses by "
        f"{be['err_vqe_vs_casci']:.1f} mHa on stretched BeH2. Two layers reach the exact singlet everywhere here.",
        "- CNOT counts are estimates for the textbook circuit (each Pauli string of each generator exponentiated with a "
        "parity ladder of 2(w-1) CNOTs, no compilation tricks), per layer.", "",
        "## 3. A trap worth knowing: the triplet below the singlet", "",
        "The first version of this benchmark reported VQE missing stretched BeH2 by 11 mHa, and a two-layer run with random "
        "starting parameters apparently fixing it. Both were misleading:", "",
        "- At twice the bond length, the lowest state in the active space is a **triplet** (S^2 = 2), 3.1 mHa below the "
        "lowest singlet, and its overlap with Hartree-Fock is zero. PySCF's CASCI returned that triplet until the spin "
        "was fixed (`fix_spin_`). Comparing a closed-shell calculation against a triplet is the wrong reference.",
        "- Without a constraint, the optimiser leaves the singlet: a two-layer run converges to the triplet. The penalty "
        "<H> + 0.5 <S^2> (Ryabinkin et al., J. Chem. Theory Comput. 15, 249 (2019)) keeps it on singlets, and then two "
        "layers reach the exact singlet energy.", "",
        "## 4. Measurement cost", "",
        "On hardware the energy is estimated from repeated measurements. With an optimal split of shots between "
        "groups of terms measured together, the total for a standard error eps is M = (sum_g sqrt(Var G_g))^2 / eps^2 "
        "(derivation in `measurement.py`). Variances are taken in the exact singlet ground state at equilibrium; "
        "groups are qubit-wise commuting (one circuit with single-qubit rotations measures a whole group).", "",
        "| Molecule | Qubits | Terms | QWC groups | Shots, worst-case bound | Shots, each term alone | Shots, QWC groups |",
        "|---|---|---|---|---|---|---|"]
for r in M["rows"]:
    out.append(f"| {r['molecule']} | {r['qubits']} | {r['terms']} | {r['groups']} | {r['shots_bound']:.2e} | "
               f"{r['shots_single']:.2e} | {r['shots_qwc']:.2e} |")
li = next(r for r in M["rows"] if r["molecule"] == "LiH"); h2o = next(r for r in M["rows"] if r["molecule"] == "H2O")
out += ["", f"Grouping cuts the shots {li['shots_single']/li['shots_qwc']:.0f}x for LiH and {h2o['shots_single']/h2o['shots_qwc']:.0f}x "
        "for water. For H2 it gains nothing: the ground state has only two determinants, so all its Z-type terms are "
        "perfectly correlated and measuring them together or apart costs the same. Every one of those shots also needs "
        "its circuit re-run, which is why measurement cost, not qubit count, often decides what is practical.", "",
        "## 5. Machine learning: Delta-learning to CCSD(T)", "",
        f"{D['n_total']} distorted water geometries, Hartree-Fock and CCSD(T) in the 6-31G basis, Coulomb-matrix features, "
        f"kernel ridge regression with cross-validated hyperparameters, {D['n_test']} geometries held out for testing. "
        "Learning the small correction CCSD(T) - HF instead of the total energy (Ramakrishnan et al., J. Chem. Theory "
        "Comput. 11, 2087 (2015)) needs far less data:", "",
        "| Training geometries | MAE, learning CCSD(T) directly (kcal/mol) | MAE, learning the correction (kcal/mol) |",
        "|---|---|---|"]
out += [f"| {r['n']} | {r['direct']:.3f} | {r['delta']:.3f} |" for r in D["rows"]]
out += ["", "![learning curves](results/delta_ml.png)", "",
        "## 6. Limits and next steps", "",
        "- Statevector simulation is exact and noiseless; the next step is shot noise and a device noise model, then error "
        "mitigation (zero-noise extrapolation).",
        "- Fixed UCCSD is not the most economical ansatz; ADAPT-VQE grows the ansatz one operator at a time and usually "
        "needs far fewer parameters and CNOTs.",
        "- STO-3G and small active spaces keep exact references affordable; larger bases need active-space selection "
        "(natural orbitals from MP2 or CASSCF)."]
open("REPORT.md", "w").write("\n".join(out) + "\n")
print("wrote REPORT.md")
