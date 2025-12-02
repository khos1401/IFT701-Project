import cuquantum
import cudaq
import math
from typing import List
from enum import Enum


class FeatureMapType(Enum):
    """ Types of feature maps available. """
    BASIC_RY = 'basic_ry'
    BASIC_RX = 'basic_rx'
    BASIC_RZ = 'basic_rz'
    BASIC_CRZ = 'basic_crz'
    BASIC_CRY = 'basic_cry'
    BASIC_CRX = 'basic_crx'
    CCRY = 'ccry'
    ZZ = 'zz'
    RY_RX = 'ry_rx'
    RY_RZ = 'ry_rz'
    U3 = 'u3'


class AnsatzType(Enum):
    """ Types of ansatz available. """
    RING = 'ring'
    ALL_TO_ALL = 'all_to_all'
    FARHI = 'farhi'
    CENTRIC = 'centric'
    TREE_TENSOR = 'tree_tensor'
    MERA = 'mera'


class QuantumCircuitConfig:
    """
    Configuration for the quantum circuit.
    """
    def __init__(
            self,
            num_features,
            n_class_qubits: int,
            num_reps: int = 2,
            feature_map_type: FeatureMapType = FeatureMapType.BASIC_RY,
            ansatz_type: AnsatzType = AnsatzType.RING,
            attention_params: int = 225,
            processing_params: int = 55,
            ):
        self.num_features = num_features
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.n_map_qubits = num_features
        self.n_class_qubits = n_class_qubits
        self.num_qubits = self.n_class_qubits + self.n_map_qubits
        self.num_reps = num_reps
        self.attention_params = attention_params
        self.processing_params = processing_params
        self.total_params = (self.attention_params + self.processing_params) * self.num_reps



###############################################################################################################################################
# Decomposed Gates for Quantum Circuits
###############################################################################################################################################

@cudaq.kernel
def cry(theta: float, control: cudaq.qubit, target: cudaq.qubit):
    """Controlled-RY gate using standard decomposition."""
    ry(theta / 2.0, target)
    x.ctrl(control, target)
    ry(-theta / 2.0, target)
    x.ctrl(control, target)

@cudaq.kernel
def crx(theta: float, control: cudaq.qubit, target: cudaq.qubit):
    """Controlled-RX gate using standard decomposition.
        HXH = Z"""
    h(target)
    rz(theta / 2.0, target)
    x.ctrl(control, target)
    rz(-theta / 2.0, target)
    x.ctrl(control, target)
    h(target)

@cudaq.kernel
def crz(theta: float, control: cudaq.qubit, target: cudaq.qubit):
    """Controlled-RZ gate using standard decomposition."""
    rz(theta / 2.0, target)
    x.ctrl(control, target)
    rz(-theta / 2.0, target)
    x.ctrl(control, target)

@cudaq.kernel
def rxx(theta: float, q1: cudaq.qubit, q2: cudaq.qubit):
    """RXX gate using standard decomposition."""
    ry(3.141592653589793 / 2.0, q1)
    ry(3.141592653589793 / 2.0, q2)
    x.ctrl(q1, q2)
    rz(theta, q2)
    x.ctrl(q1, q2)
    ry(-3.141592653589793 / 2.0, q1)
    ry(-3.141592653589793 / 2.0, q2)

@cudaq.kernel
def rzz(theta: float, q1: cudaq.qubit, q2: cudaq.qubit):
    """RZZ gate using standard decomposition."""
    x.ctrl(q1, q2)
    rz(theta, q2)
    x.ctrl(q1, q2)

@cudaq.kernel
def ccry(theta: float, control1: cudaq.qubit, control2: cudaq.qubit, target: cudaq.qubit):
    """Controlled-Controlled-RY gate using standard decomposition."""
    ry(theta / 4.0, target)
    x.ctrl(control2, target)
    ry(-theta / 4.0, target)
    x.ctrl(control2, target)

    x.ctrl(control1, control2)

    ry(-theta / 4.0, target)
    x.ctrl(control2, target)
    ry(theta / 4.0, target)
    x.ctrl(control2, target)

    x.ctrl(control1, control2)

    ry(theta / 4.0, target)
    x.ctrl(control1, target)
    ry(-theta / 4.0, target)
    x.ctrl(control1, target)

###############################################################################################################################################
# Feature Map Implementations
###############################################################################################################################################

# NOTE: helpers accept a cudaq.qview `q` allocated in the top-level kernel and
# never allocate qubits themselves. Keep indices within [0, len(q)-1].

@cudaq.kernel
def ry_feature_map(q: cudaq.qview,
                   n_class_qubits: int,
                   n_map_qubits: int,
                   input_data: List[float],
                   ):
    """Simple RY feature map encoding (acts on map qubits)."""
    h(q)  # optional Hadamard layer before RY encoding
    for i in range(n_map_qubits):
        if i < len(input_data):
            ry(input_data[i], q[n_class_qubits + i])


###############################################################################################################################################
# Ansatz Implementations (NO qvector allocation here)
###############################################################################################################################################

@cudaq.kernel
def ring_ansatz(q: cudaq.qview,
                n_class_qubits: int,
                n_map_qubits: int,
                attention_params: List[float],
                processing_params: List[float],
                num_layers: int = 2,
                ):
    """Basic ring-entangling ansatz operating on the provided qubits,
    using explicit class/map start indices for stable positioning.
    """

    # Attention
    a = 0
    for k in range(n_map_qubits):
        idx = n_class_qubits + k
        if a < len(attention_params):
            ry(attention_params[a], q[idx]); a += 1
        if a < len(attention_params):
            rz(attention_params[a], q[idx]); a += 1

    for k in range(n_map_qubits):
        idx = n_class_qubits + k
        if a < len(attention_params):
            ry(attention_params[a], q[idx]); a += 1

    # Entangling attention (2 options; A or B)
    if n_map_qubits >= 2:
        # (A) ring entanglement
        for k in range(n_map_qubits):
            c = n_class_qubits + k
            t = n_class_qubits + ((k + 1) % n_map_qubits)
            if a < len(attention_params):
                cry(attention_params[a], q[c], q[t]); a += 1

    # Processing Layers
    p = 0
    L = num_layers
    if L <= 0:
        L = 1

    for _ in range(L):
        # RY on map qubits
        for k in range(n_map_qubits):
            idx = n_class_qubits + k
            if p < len(processing_params):
                ry(processing_params[p], q[idx]); p += 1

        # CX from map k to class k, up to min counts
        mcc = n_class_qubits
        if n_map_qubits < mcc:
            mcc = n_map_qubits
        for c in range(mcc):
            x.ctrl(q[n_class_qubits + c], q[c])

        # Optional RZ on class qubits
        for c in range(n_class_qubits):
            if p < len(processing_params):
                rz(processing_params[p], q[c]); p += 1


###############################################################################################################################################
# Final Circuit/Kernel Constructor (single allocation point)
###############################################################################################################################################

@cudaq.kernel
def kernel_constructor(n_qubits: int,
                       n_class_qubits: int,
                       n_map_qubits: int,
                       FEATURE_MAP: int,
                       ANSATZ: int,
                       input_params: List[float],
                       attention_params: List[float],
                       processing_params: List[float],
                       num_layers: int = 2):
    """Constructs the full quantum circuit kernel with specified feature map and ansatz.
    0: BASIC_RY feature map
    0: RING ansatz
    """

    # Single allocation point
    q = cudaq.qvector(n_qubits)

    # --- feature map ---
    if FEATURE_MAP == 0:
        ry_feature_map(q, n_class_qubits, n_map_qubits, input_params)
        

    # --- ansatz ---
    if ANSATZ == 0:
        ring_ansatz(q, n_class_qubits, n_map_qubits, attention_params, processing_params, num_layers)