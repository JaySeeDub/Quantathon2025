# shadows.py
"""
Classical Shadows utilities for measurement-based feature embedding.

This module supports:
  - Random local Pauli (X/Y/Z) measurement rounds (Pauli-3 scheme)
  - Estimating many Pauli-string expectations from the same measurement pool
  - Ready-made feature banks (singles, ring pairs, all weight-2)
  - Building an (N_samples x N_features) feature matrix from circuits

References:
  - Huang, Kueng, Preskill, "Predicting many properties of a quantum state from
    few measurements", Nature Physics 16, 1050–1057 (2020).
    (Pauli-3 randomized local measurements; many-observable estimation)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Tuple, Dict, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ----------------------------
# Public configuration object
# ----------------------------
@dataclass
class ShadowConfig:
    """Configuration for collecting random-basis measurements."""
    T: int = 200                 # number of random-basis rounds per sample
    shots: int = 1024            # shots per round
    seed: Optional[int] = 1234   # seed for simulator + RNG (None => unseeded)
    backend: Optional[object] = None  # qiskit backend (defaults to AerSimulator)


# ----------------------------
# Random basis generation
# ----------------------------
def random_bases(n: int, rng: np.random.Generator) -> List[str]:
    """Sample a random local basis in {X,Y,Z}^n."""
    return rng.choice(["X", "Y", "Z"], size=n).tolist()


def add_measurement_layer(base: QuantumCircuit, bases: Sequence[str]) -> QuantumCircuit:
    """
    Compose `base` with per-qubit rotations to measure in the requested bases,
    then measure in the computational basis.
    """
    n = base.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(base, inplace=True)
    for q, b in enumerate(bases):
        if b == "X":
            qc.h(q)            # X-basis via H then Z-measure
        elif b == "Y":
            qc.sdg(q); qc.h(q) # Y-basis via HS^† then Z-measure
        # if Z: do nothing
        qc.measure(q, q)
    return qc


# ----------------------------
# Collect shadows (measurements)
# ----------------------------
def collect_shadows(
    stateprep: QuantumCircuit,
    cfg: ShadowConfig = ShadowConfig(),
) -> Tuple[List[List[str]], List[np.ndarray]]:
    """
    Run T random local-basis rounds on `stateprep`.

    Returns
    -------
    bases_list : list length T
        Each element is a length-n list of 'X'/'Y'/'Z' strings.
    outcomes : list length T
        Each element is a (shots, n) array of ±1 measurement outcomes.
        Bit order is q0..q(n-1) across columns.
    """
    n = stateprep.num_qubits
    rng = np.random.default_rng(cfg.seed)
    bases_list = [random_bases(n, rng) for _ in range(cfg.T)]

    # Prepare backend
    backend = cfg.backend or AerSimulator(seed_simulator=cfg.seed)

    # Build circuits for all rounds
    circuits = [add_measurement_layer(stateprep, b) for b in bases_list]

    # Execute batched
    job = backend.run(circuits, shots=cfg.shots)
    result = job.result()

    outcomes: List[np.ndarray] = []
    for t in range(cfg.T):
        counts = result.get_counts(t)  # dict: bitstring -> count
        # Convert to matrix of ±1 with qubit columns in order q0..q(n-1)
        mats = []
        for bitstr, c in counts.items():
            # Qiskit returns classical bits c_(n-1)..c_0 in the string left..right.
            # We measured q -> c with same indices, so reverse to map to q0..q(n-1).
            bits = np.fromiter(bitstr[::-1], dtype=np.uint8)  # '0'/'1' bytes
            bits = bits - ord('0')                              # 0/1 ints
            pm = 1 - 2*bits                                    # 0 -> +1, 1 -> -1
            mats.append(np.tile(pm, (c, 1)))
        outcomes.append(np.vstack(mats) if mats else np.zeros((0, n), dtype=int))

    return bases_list, outcomes


# ----------------------------
# Pauli-string utilities
# ----------------------------
Pauli = Tuple[str, ...]  # e.g., ('Z','I','X',...)


def paulis_singles_xyz(n: int) -> List[Pauli]:
    """All single-qubit Paulis {X,Y,Z} on each of n qubits (size 3n)."""
    out: List[Pauli] = []
    for i in range(n):
        for a in ("X", "Y", "Z"):
            s = ["I"] * n
            s[i] = a
            out.append(tuple(s))
    return out


def paulis_ring_pairs(n: int, axes: Tuple[str, str] = ("Z", "Z")) -> List[Pauli]:
    """Ring pairs (i, i+1 mod n) with specified axes (e.g., ('Z','Z') => ZZ ring)."""
    out: List[Pauli] = []
    for i in range(n):
        j = (i + 1) % n
        s = ["I"] * n
        s[i], s[j] = axes
        out.append(tuple(s))
    return out


def paulis_all_weight2(n: int, axes: Sequence[str] = ("X", "Y", "Z")) -> List[Pauli]:
    """All unordered pairs i<j with all axis combinations from `axes`."""
    out: List[Pauli] = []
    for i in range(n):
        for j in range(i + 1, n):
            for a in axes:
                for b in axes:
                    s = ["I"] * n
                    s[i], s[j] = a, b
                    out.append(tuple(s))
    return out  # size: C(n,2) * len(axes)^2  (e.g., 28 * 9 = 252 for n=8)


def label_of(P: Pauli) -> str:
    """Compact string label like 'ZIZX...'. Useful for column names."""
    return "".join(P)


# ----------------------------
# Estimation (classical shadows)
# ----------------------------
def estimate_pauli_expectations(
    bases_list: Sequence[Sequence[str]],
    outcomes: Sequence[np.ndarray],
    pauli_list: Sequence[Pauli],
) -> Dict[Pauli, float]:
    """
    Estimate <P> for each Pauli string P using Pauli-3 classical shadows.

    For a Pauli string P of weight w, a round is "compatible" iff every
    non-identity qubit of P was measured in the same axis. On compatible
    rounds, we multiply the ±1 outcomes on the support and scale by 3^w.
    Incompatible rounds contribute 0. Average across rounds.

    This yields an *unbiased* estimator for Pauli expectations under the
    Pauli-3 measurement scheme.

    Returns
    -------
    dict mapping P -> expectation estimate
    """
    assert len(bases_list) == len(outcomes), "Mismatched bases and outcomes."
    T = len(bases_list)
    n = len(bases_list[0]) if T > 0 else 0

    # Precompute supports / weights
    supports: List[List[int]] = []
    weights: List[int] = []
    for P in pauli_list:
        supp = [i for i, a in enumerate(P) if a != "I"]
        supports.append(supp)
        weights.append(len(supp))

    feats: Dict[Pauli, float] = {}
    for idx, P in enumerate(pauli_list):
        supp = supports[idx]
        w = weights[idx]
        if w == 0:
            feats[P] = 1.0
            continue
        scale = 3 ** w

        acc = 0.0
        # Average over T rounds (incompatible ⇒ implicit 0 contribution)
        for t in range(T):
            bases = bases_list[t]
            if not all(bases[q] == P[q] for q in supp):
                continue
            m = outcomes[t][:, supp]  # (shots, w) of ±1
            if m.size == 0:
                continue
            prod = np.prod(m, axis=1)        # (shots,)
            acc += scale * float(np.mean(prod))
        feats[P] = acc / max(T, 1)

    return feats


# ----------------------------
# High-level: circuits -> matrix
# ----------------------------
def build_feature_matrix_from_circuits(
    circuits: Sequence[QuantumCircuit],
    pauli_list: Sequence[Pauli],
    cfg: ShadowConfig = ShadowConfig(),
) -> np.ndarray:
    """
    For each circuit, collect random-basis measurements and estimate features.
    Returns an array of shape (N_samples, N_features) in the order of `pauli_list`.
    """
    N = len(circuits)
    F = len(pauli_list)
    X = np.zeros((N, F), dtype=float)
    for k, circ in enumerate(circuits):
        bases, outs = collect_shadows(circ, cfg)
        feats = estimate_pauli_expectations(bases, outs, pauli_list)
        X[k, :] = [feats[P] for P in pauli_list]
    return X


# ----------------------------
# Example usage (self-test)
# ----------------------------
if __name__ == "__main__":
    # Tiny smoke test on a trivial 3-qubit state
    from qiskit import QuantumCircuit

    n = 3
    qc = QuantumCircuit(n)
    qc.h(0); qc.cx(0, 1); qc.h(2)  # some entanglement + superposition

    cfg = ShadowConfig(T=100, shots=512, seed=7)

    # Choose some features: singles + all weight-2 ZZ
    singles = paulis_singles_xyz(n)
    zz_pairs = paulis_all_weight2(n, axes=("Z",))  # only ZZ, all pairs
    paulis = singles + zz_pairs

    bases, outs = collect_shadows(qc, cfg)
    feats = estimate_pauli_expectations(bases, outs, paulis)

    # Pretty-print a few features
    names = [label_of(P) for P in paulis[:10]]
    vals = [feats[P] for P in paulis[:10]]
    print("First 10 features:")
    for name, val in zip(names, vals):
        print(f"  {name:>6s}  ->  {val:+.4f}")
