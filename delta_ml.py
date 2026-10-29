"""Machine learning for chemistry: learn CCSD(T) energies of distorted water molecules from their geometry, either
directly or as a correction to a cheap Hartree-Fock calculation ("Delta-learning", Ramakrishnan et al., J. Chem.
Theory Comput. 11, 2087 (2015)). Coulomb-matrix features (Rupp et al. 2012), kernel ridge regression, learning curves."""
import json
import pathlib
import numpy as np
from pyscf import gto, scf, cc
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KCAL = 627.5095
CACHE = pathlib.Path("results/water_energies.json")
Z = np.array([8, 1, 1])


def water(r1, r2, ang):
    a = np.radians(ang)
    return np.array([[0, 0, 0], [r1, 0, 0], [r2 * np.cos(a), r2 * np.sin(a), 0]])


def data(n=300, seed=7):
    if CACHE.exists():
        d = json.loads(CACHE.read_text())
        return np.array(d["xyz"]), np.array(d["hf"]), np.array(d["ccsdt"])
    rng = np.random.default_rng(seed)
    xyz, hf, cct = [], [], []
    for _ in range(n):
        g = water(rng.uniform(0.85, 1.25), rng.uniform(0.85, 1.25), rng.uniform(90, 125))
        mol = gto.M(atom=[("O", g[0]), ("H", g[1]), ("H", g[2])], basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        c = cc.CCSD(mf).run()
        xyz.append(g.tolist()); hf.append(mf.e_tot); cct.append(c.e_tot + c.ccsd_t())
    CACHE.write_text(json.dumps(dict(xyz=xyz, hf=hf, ccsdt=cct)))
    return np.array(xyz), np.array(hf), np.array(cct)


def coulomb(g):
    d = np.linalg.norm(g[:, None] - g[None], axis=-1) / 0.529177          # bohr
    M = np.outer(Z, Z) / np.where(d > 0, d, 1)
    return M[np.triu_indices(3, 1)]                                        # off-diagonal: the diagonal never changes


xyz, e_hf, e_cc = data()
X = np.array([coulomb(g) for g in xyz])
y_direct, y_delta = e_cc * KCAL, (e_cc - e_hf) * KCAL
ntr = 220
Xtr, Xte = X[:ntr], X[ntr:]
mu, sd = Xtr.mean(0), Xtr.std(0)
grid = {"alpha": [1e-10, 1e-8, 1e-6, 1e-4], "gamma": [0.01, 0.03, 0.1, 0.3, 1.0]}
rows = []
print(f"{len(X)} geometries; test set {len(X) - ntr}. Spread of the test energies: direct "
      f"{np.std(y_direct[ntr:]):.1f} kcal/mol, correlation correction {np.std(y_delta[ntr:]):.2f} kcal/mol")
print(" N train   MAE direct   MAE Delta (kcal/mol)")
for n in (10, 20, 40, 80, 160, 220):
    out = {}
    for name, y in (("direct", y_direct), ("delta", y_delta)):
        m = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=min(5, n // 2)).fit((Xtr[:n] - mu) / sd, y[:n] - y[:n].mean())
        pred = m.predict((Xte - mu) / sd) + y[:n].mean()
        out[name] = float(np.abs(pred - y[ntr:]).mean())
    rows.append(dict(n=n, **out))
    print(f"  {n:5d}     {out['direct']:8.3f}    {out['delta']:8.3f}")
need = lambda key: next((r["n"] for r in rows if r[key] < 1.0), None)
print(f"training geometries needed for < 1 kcal/mol: direct {need('direct')}, Delta {need('delta')}")
json.dump(dict(rows=rows, n_total=len(X), n_test=len(X) - ntr), open("results/delta_ml.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.loglog([r["n"] for r in rows], [r["direct"] for r in rows], "o-", label="learn CCSD(T) directly")
ax.loglog([r["n"] for r in rows], [r["delta"] for r in rows], "s-", label="learn CCSD(T) - HF (Delta)")
ax.axhline(1.0, color="r", lw=0.7, ls=":", label="1 kcal/mol")
ax.set(xlabel="training geometries", ylabel="test MAE (kcal/mol)", title="Water, 6-31G: learning curves")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("results/delta_ml.png", dpi=140)
