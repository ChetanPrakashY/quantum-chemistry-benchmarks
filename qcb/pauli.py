"""Pauli strings in the binary (symplectic) form: a string on n qubits is two n-bit integers x and z, meaning
i^(x.z) X^x Z^z, so Y = i X Z. Products, sums and sparse matrices without ever building 2^n x 2^n dense matrices."""
import numpy as np
import scipy.sparse as sp


def popcount(v):
    return bin(v).count("1")


def mul(p1, p2):
    """(x1, z1) * (x2, z2) -> (phase power of i, (x3, z3))."""
    (x1, z1), (x2, z2) = p1, p2
    x3, z3 = x1 ^ x2, z1 ^ z2
    e = popcount(x1 & z1) + popcount(x2 & z2) - popcount(x3 & z3) + 2 * popcount(z1 & x2)
    return e % 4, (x3, z3)


def op_mul(A, B):
    """Product of two operators, each a dict {(x, z): coefficient}."""
    out = {}
    for pa, ca in A.items():
        for pb, cb in B.items():
            e, p = mul(pa, pb)
            out[p] = out.get(p, 0) + ca * cb * (1j ** e)
    return out


def op_add(A, B, scale=1.0):
    out = dict(A)
    for p, c in B.items():
        out[p] = out.get(p, 0) + scale * c
    return out


def clean(A, tol=1e-12):
    return {p: c for p, c in A.items() if abs(c) > tol}


def label(p, n):
    x, z = p
    return "".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in range(n))


def weight(p):
    return popcount(p[0] | p[1])


def to_sparse(A, n):
    """Sparse matrix: P(x, z)|b> = i^(x.z) (-1)^(z.b) |b xor x>."""
    dim = 2 ** n
    b = np.arange(dim)
    M = sp.csr_matrix((dim, dim), dtype=complex)
    for (x, z), c in A.items():
        parity = np.array([popcount(int(v)) & 1 for v in (b & z)])
        vals = c * (1j ** popcount(x & z)) * (1 - 2 * parity)
        M = M + sp.csr_matrix((vals, (b ^ x, b)), shape=(dim, dim))
    return M


def jw_annihilation(j):
    """Jordan-Wigner a_j = Z_0 ... Z_{j-1} (X_j + i Y_j)/2: a sum of two Pauli strings. (X + iY)/2 = |0><1|,
    which empties qubit j; the Z string gives the fermionic sign."""
    zs = (1 << j) - 1
    return {(1 << j, zs): 0.5, (1 << j, zs | (1 << j)): 0.5j}


def dagger(A):
    return {p: np.conj(c) for p, c in A.items()}       # Pauli strings are Hermitian
