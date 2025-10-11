# -*- coding: utf-8 -*-
# End-to-end QNN training loop (Qiskit Aer Estimator + SPSA), 8-qubit covariance-driven ansatz
# - Data encoding: one pass RZ(x_i) -> RX(x_i)
# - Trainable params: per-layer RY on each qubit + per-edge RZZ angles
# - Readout: mean Z expectation across all qubits -> sigmoid -> BCE loss
#
# Requirements: qiskit >= 1.0, qiskit-aer >= 0.14, pandas

import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
import pandas as pd


# ---- Aer Estimator (GPU if available) ----
backend_options = {
    "device": "GPU",  # change to "CPU" if your build lacks GPU
}
est = Estimator(backend_options=backend_options)

# ---------- Utilities: covariance -> correlation graph ----------
def covariance_to_corr(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(C))
    # protect against zeros on diagonal
    d = np.where(d == 0.0, 1e-12, d)
    R = C / np.outer(d, d)
    np.fill_diagonal(R, 0.0)
    return np.clip(R, -1.0, 1.0)

def sparsify_corr(R: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    M = R.copy()
    M[np.abs(M) < threshold] = 0.0
    np.fill_diagonal(M, 0.0)
    return M

def edge_list(W: np.ndarray) -> List[Tuple[int, int, float]]:
    n = W.shape[0]
    return [(i, j, W[i, j]) for i, j in combinations(range(n), 2) if W[i, j] != 0.0]


# ---------- Model/Params container ----------
@dataclass
class QNN:
    circuit: QuantumCircuit
    n: int
    layers: int
    edges: List[Tuple[int, int, float]]
    # data parameters (bound per-sample, not optimized)
    phi_z: ParameterVector         # length n
    phi_x: ParameterVector         # length n
    # trainable parameters
    theta_ry: List[ParameterVector]                 # per-layer length n
    theta_edge: Dict[Tuple[int, int, int], Parameter]  # (ell,i,j) -> Parameter


# ---------- Build the QNN ----------
def build_qnn(C: np.ndarray, layers: int = 2, edge_threshold: float = 0.2) -> QNN:
    """
    U(x; theta) for n=len(C):
      - Encode x with ∏_i RZ(x_i) RX(x_i)
      - For each layer:
          ∏_i RY(theta_ry[ell,i])
          ∏_(i<j) RZZ(theta_edge[ell,i,j] * w_ij)
    Readout: mean Z expectation over all qubits (done outside circuit).
    """
    n = C.shape[0]
    R = covariance_to_corr(C)
    W = sparsify_corr(R, edge_threshold)
    E = edge_list(W)

    qc = QuantumCircuit(n, name="CovGraphQNN")

    # Data parameters (bound per sample)
    phi_z = ParameterVector("phi_z", n)
    phi_x = ParameterVector("phi_x", n)
    for q in range(n):
        qc.rz(phi_z[q], q)
        qc.rx(phi_x[q], q)

    # Trainable parameters
    theta_ry: List[ParameterVector] = []
    theta_edge: Dict[Tuple[int, int, int], Parameter] = {}
    for ell in range(layers):
        pv = ParameterVector(f"theta_ry_l{ell}", n)
        theta_ry.append(pv)
        # single-qubit trainables
        for q in range(n):
            qc.ry(pv[q], q)
        # edge-wise ZZ
        for (i, j, w) in E:
            p = Parameter(f"theta_e_l{ell}_{i}_{j}")
            theta_edge[(ell, i, j)] = p
            qc.append(RZZGate(p * w), [i, j])

    return QNN(
        circuit=qc,
        n=n,
        layers=layers,
        edges=E,
        phi_z=phi_z,
        phi_x=phi_x,
        theta_ry=theta_ry,
        theta_edge=theta_edge,
    )


# ---------- Readout observable: mean Z over all qubits ----------
def mean_z_observable(n: int) -> SparsePauliOp:
    # (1/n) * sum_i Z_i  == average magnetization
    paulis = []
    coeffs = []
    for i in range(n):
        z_str = ["I"] * n
        z_str[i] = "Z"
        # Reverse because rightmost char applies to qubit 0 in Qiskit little-endian convention
        paulis.append("".join(reversed(z_str)))
        coeffs.append(1.0 / n)
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


# ---------- Loss & metrics ----------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def binary_cross_entropy(logit, y):
    p = sigmoid(logit)
    eps = 1e-10
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

def accuracy(logits, y):
    preds = (sigmoid(logits) >= 0.5).astype(int)
    return float(np.mean(preds == y))


# ---------- Parameter packing helpers ----------
def init_theta(qnn: QNN, seed: int = 7, scale: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # concatenate [all theta_ry; all theta_edge]
    ry_total = qnn.layers * qnn.n
    edge_total = len(qnn.edges) * qnn.layers
    theta = np.zeros(ry_total + edge_total)
    theta[:ry_total] = rng.normal(0, scale, size=ry_total)
    theta[ry_total:] = rng.normal(0, scale, size=edge_total)
    return theta

def unpack_theta(qnn: QNN, theta: np.ndarray):
    """Return dict mapping Parameter -> value."""
    bind = {}
    idx = 0
    # theta_ry
    for ell in range(qnn.layers):
        for q in range(qnn.n):
            bind[qnn.theta_ry[ell][q]] = float(theta[idx]); idx += 1
    # theta_edge in fixed (ell, i, j) order for reproducibility
    for ell in range(qnn.layers):
        for (i, j, _) in qnn.edges:
            bind[qnn.theta_edge[(ell, i, j)]] = float(theta[idx]); idx += 1
    return bind

def data_binding(qnn: QNN, x: np.ndarray):
    """Bind phi_z_i=x_i, phi_x_i=x_i"""
    bind = {}
    for i in range(qnn.n):
        bind[qnn.phi_z[i]] = float(x[i])
        bind[qnn.phi_x[i]] = float(x[i])
    return bind


# ---------- Forward pass ----------
def forward_logits(estimator: Estimator, qnn: QNN, obs: SparsePauliOp,
                   X: np.ndarray, theta: np.ndarray, shots: int = None) -> np.ndarray:
    """
    Returns model logits for each x in X:
      logit(x) = <mean Z> (no extra linear head; simple and stable)
    """
    theta_bind = unpack_theta(qnn, theta)
    circuits = []
    observables = []
    for x in X:
        param_bind = {**theta_bind, **data_binding(qnn, x)}
        circuits.append(qnn.circuit.assign_parameters(param_bind))
        observables.append(obs)

    run_opts = {} if shots is None else {"shots": shots}
    res = estimator.run(
        circuits=circuits,
        observables=observables,
        parameter_values=None,
        **run_opts
    ).result()
    logits = np.array(res.values, dtype=float)
    return logits


# ---------- SPSA optimizer ----------
@dataclass
class SPSAConfig:
    maxiter: int = 200
    a: float = 0.05
    c: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101
    seed: int = 123

def spsa_schedule(cfg: SPSAConfig, k: int):
    ak = cfg.a / ((k + 1) ** cfg.alpha)
    ck = cfg.c / ((k + 1) ** cfg.gamma)
    return ak, ck

def spsa_step(theta: np.ndarray, grad_est: np.ndarray, ak: float):
    return theta - ak * grad_est

def estimate_gradient_spsa(estimator: Estimator, qnn: QNN, obs: SparsePauliOp,
                           X: np.ndarray, y: np.ndarray, theta: np.ndarray,
                           ck: float, rng: np.random.Generator, shots: int = None) -> np.ndarray:
    # Bernoulli ±1 perturbation
    delta = rng.choice([-1.0, 1.0], size=theta.shape)
    theta_plus  = theta + ck * delta
    theta_minus = theta - ck * delta

    # Forward passes
    logits_plus  = forward_logits(estimator, qnn, obs, X, theta_plus,  shots=shots)
    logits_minus = forward_logits(estimator, qnn, obs, X, theta_minus, shots=shots)

    # Losses
    Lp = np.mean([binary_cross_entropy(lp, yi) for lp, yi in zip(logits_plus,  y)])
    Lm = np.mean([binary_cross_entropy(lm, yi) for lm, yi in zip(logits_minus, y)])

    # SPSA gradient estimate (element-wise)
    ghat = (Lp - Lm) / (2.0 * ck) * (1.0 / delta)
    return ghat


# ---------- Training loop ----------
def train_qnn(C: np.ndarray, X: np.ndarray, y: np.ndarray,
              layers: int = 2, edge_threshold: float = 0.2,
              shots: int = None,  # set to an int (e.g., 4000) for sampling; None -> exact
              spsa_cfg: SPSAConfig = SPSAConfig()):
    """
    Returns: (theta_best, history, qnn, obs)
    """
    assert X.shape[1] == C.shape[0], "Feature dimension must match covariance size."
    assert set(np.unique(y)).issubset({0, 1}), "Binary labels expected (0/1)."

    qnn = build_qnn(C, layers=layers, edge_threshold=edge_threshold)
    obs = mean_z_observable(qnn.n)

    # Init params
    theta = init_theta(qnn, seed=7)
    rng = np.random.default_rng(spsa_cfg.seed)
    best_theta = theta.copy()

    # Track
    history = {"loss": [], "acc": []}

    for k in range(spsa_cfg.maxiter):
        ak, ck = spsa_schedule(spsa_cfg, k)

        # Grad estimate
        ghat = estimate_gradient_spsa(est, qnn, obs, X, y, theta, ck, rng, shots=shots)

        # Step
        theta = spsa_step(theta, ghat, ak)

        # Eval
        logits = forward_logits(est, qnn, obs, X, theta, shots=shots)
        loss = float(np.mean([binary_cross_entropy(li, yi) for li, yi in zip(logits, y)]))
        accu = accuracy(logits, y)

        history["loss"].append(loss)
        history["acc"].append(accu)

        # Keep best by loss
        if loss <= min(history["loss"]):
            best_theta = theta.copy()

        # Simple progress print (every 10 iters)
        if (k + 1) % 10 == 0 or k == 0:
            print(f"iter {k+1:4d} | loss {loss:.4f} | acc {accu:.3f}")

    return best_theta, history, qnn, obs


# ---------- Load training data ----------
df = pd.read_csv("../Data/correlation_matrix.csv")
df = df.drop(df.columns[0], axis=1)
data = df.to_numpy()

df = pd.read_csv("../Data/X_train_scaled.csv")
df = df.drop(df.columns[0], axis=1)
X = df.to_numpy()

df = pd.read_excel("../Data/2025-Quantathon-Tornado-Q-training_data-640-examples.xlsx")
Y = df["ef_binary"].to_numpy().astype(int)

C = data

# ---------- Train ----------
best_theta, hist, qnn, obs = train_qnn(
    C, X, Y, layers=5, edge_threshold=0.25,
    shots=None,  # set to an int to mimic hardware sampling
    spsa_cfg=SPSAConfig(maxiter=200, a=0.08, c=0.15),
)


# ---- Validation / Test ----
VAL_CSV  = "../Data/X_val_scaled.csv"
TEST_CSV = "../Data/X_test_scaled.csv"
FEATURE_COLS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]  # or None to use all-but-last
LABEL_COL = "label"  # set to your actual label column if present

def load_xy(csv_path, feature_cols, label_col):
    df = pd.read_csv(csv_path)
    if feature_cols is None:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy().astype(int) if (label_col in df.columns) else None
    else:
        X = df[feature_cols].to_numpy()
        y = df[label_col].to_numpy().astype(int) if (label_col in df.columns) else None
    return X, y

# ---- Load val (already scaled) ----
Xva, yva = load_xy(VAL_CSV, FEATURE_COLS, LABEL_COL)

# ---- Inference on validation ----
val_logits = forward_logits(est, qnn, obs, Xva, best_theta, shots=None)
val_probs  = 1.0 / (1.0 + np.exp(-val_logits))
val_pred   = (val_probs >= 0.5).astype(int)

# ---- Metrics (requires labels present in val CSV) ----
if yva is not None:
    acc = float(np.mean(val_pred == yva))
    tp = int(np.sum((val_pred == 1) & (yva == 1)))
    fp = int(np.sum((val_pred == 1) & (yva == 0)))
    fn = int(np.sum((val_pred == 0) & (yva == 1)))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    print(f"Validation | acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
else:
    print("Validation labels not found; printed predictions only.")
    print("Validation predictions (first 10):", val_pred[:10].tolist())

# ---- Optional: Test set (no labels expected) ----
if TEST_CSV is not None:
    Xte, yte = load_xy(TEST_CSV, FEATURE_COLS, LABEL_COL)  # yte may be None
    test_logits = forward_logits(est, qnn, obs, Xte, best_theta, shots=None)
    test_probs  = 1.0 / (1.0 + np.exp(-test_logits))
    test_pred   = (test_probs >= 0.5).astype(int)
    print("Test predictions (first 10):", test_pred[:10].tolist())
