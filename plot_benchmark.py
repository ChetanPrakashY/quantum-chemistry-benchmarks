"""Draw results/benchmark.png from results/benchmark.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = json.load(open("results/benchmark.json"))
FLOOR = 1e-6                                        # errors below this are numerical precision: drawn on the floor
mols = list(dict.fromkeys(r["molecule"] for r in B["rows"]))
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for k, name in enumerate(mols):
    rr = [r for r in B["rows"] if r["molecule"] == name]
    xs = [r["scale"] for r in rr]
    for key, lab, st in [("err_hf", "HF", "o-"), ("err_mp2", "MP2", "o-"), ("err_ccsd", "CCSD", "o-"),
                         ("err_ccsd_t", "CCSD(T)", "o-"), ("err_vqe_vs_casci", "UCCSD-VQE, 1 layer (vs CASCI)", "k*--"),
                         ("err_vqe2_vs_casci", "UCCSD-VQE, 2 layers (vs CASCI)", "kx:")]:
        ax[k].semilogy(xs, [max(abs(r[key]), FLOOR) for r in rr], st, ms=4, label=lab)
    ax[k].axhline(B["chemical_accuracy_mha"], color="r", lw=0.7, ls=":", label="1 kcal/mol")
    ax[k].axhspan(FLOOR * 0.5, FLOOR * 1.5, color="0.9")
    ax[k].set(title=f"{name}: VQE in {rr[0]['active']}, {rr[0]['qubits']} qubits", xlabel="bond length / equilibrium")
    ax[k].title.set_fontsize(9)
ax[0].set_ylabel("|error| (mHa); grey band: < 1e-6")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("results/benchmark.png", dpi=140)
