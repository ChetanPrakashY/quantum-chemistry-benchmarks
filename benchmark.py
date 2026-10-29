"""Benchmark: classical methods against exact FCI in the full STO-3G space, and UCCSD-VQE against exact CASCI in an
active space, for four molecules at equilibrium and stretched bond lengths. Writes results/benchmark.json and
results/benchmark.png."""
import json
import time
import numpy as np
from pyscf import mp, cc, fci
from qcb.pauli import to_sparse
from qcb.hamiltonian import active_space, qubit_hamiltonian, hf_state, s_squared
from qcb.vqe import uccsd_generators, run_vqe, cnot_estimate
from benchmark_geoms import geom, ACTIVE

MHA = 1000.0
CHEM_ACC = 1.594                                            # 1 kcal/mol in mHa


rows = []
for name in ACTIVE:
    for s in (1.0, 1.5, 2.0):
        t0 = time.time()
        ncas, ne = ACTIVE[name]
        a = active_space(geom(name, s), "sto-3g", ncas, ne)
        mf = a["mf"]
        solver = fci.FCI(mf)
        fci.addons.fix_spin_(solver, ss=0)                  # singlet: the state RHF, CCSD and UCCSD describe
        e_fci = solver.kernel()[0]
        e_mp2 = mp.MP2(mf).run().e_tot
        ccsd = cc.CCSD(mf); ccsd.max_cycle = 200; ccsd.run()
        e_ccsd_t = ccsd.e_tot + ccsd.ccsd_t()
        Hd = qubit_hamiltonian(a["e_core"], a["h1"], a["h2"])
        nq = 2 * ncas
        gens, labels = uccsd_generators(nq, ne)
        Hs, S2 = to_sparse(Hd, nq), to_sparse(s_squared(nq), nq)
        G = [to_sparse(g, nq) for g in gens]
        vqe = {}
        for layers in (1, 2):                       # spin penalty 0.5 S^2 keeps the optimiser on singlets
            e, theta, psi, nit = run_vqe(Hs, G, hf_state(nq, ne), layers=layers, penalty=0.5 * S2, starts=3)
            vqe[layers] = ((e - a["e_casci"]) * MHA, np.vdot(psi, S2 @ psi).real, int(nit))
        e_vqe = a["e_casci"] + vqe[1][0] / MHA; nit = vqe[1][2]
        r = dict(molecule=name, scale=s, qubits=nq, active=f"({ne}e,{ncas}o)", pauli_terms=len(Hd), params=len(gens),
                 cnots=cnot_estimate(gens), iterations=int(nit),
                 err_hf=(a["e_hf"] - e_fci) * MHA, err_mp2=(e_mp2 - e_fci) * MHA, err_ccsd=(ccsd.e_tot - e_fci) * MHA,
                 err_ccsd_t=(e_ccsd_t - e_fci) * MHA, ccsd_converged=bool(ccsd.converged),
                 err_vqe_vs_casci=vqe[1][0], err_vqe2_vs_casci=vqe[2][0], s2_vqe=vqe[1][1], s2_vqe2=vqe[2][1], casci_vs_fci=(a["e_casci"] - e_fci) * MHA,
                 e_fci=e_fci, seconds=time.time() - t0)
        rows.append(r)
        print(f"{name:5} x{s:.1f}  {nq:2d}q {len(Hd):4d} terms {len(gens):3d} params | errors vs FCI (mHa): "
              f"HF {r['err_hf']:7.2f} MP2 {r['err_mp2']:7.2f} CCSD {r['err_ccsd']:7.3f} CCSD(T) {r['err_ccsd_t']:7.3f}"
              f" | VQE-CASCI 1 layer {r['err_vqe_vs_casci']:.1e}, 2 layers {r['err_vqe2_vs_casci']:.1e} mHa | {r['seconds']:.0f}s")
json.dump(dict(chemical_accuracy_mha=CHEM_ACC, rows=rows), open("results/benchmark.json", "w"), indent=1)

import plot_benchmark  # noqa: E402,F401  (draws results/benchmark.png from the JSON just written)
