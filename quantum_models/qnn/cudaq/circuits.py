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

@cudaq.kernel
def y_rot(q:cudaq.qubit, theta:float):
    ry(2.0*theta, q)


###############################################################################################################################################
# MCQRI Feature Map Implementation
###############################################################################################################################################

@cudaq.kernel
def frqi(q_pos:int, angles:list[float], index_q:list[int], num_x:list[int]):
    qubits = cudaq.qvector(q_pos)
    aux = cudaq.qubit()

    h(qubits)

    j = 0
    count = 0
    for theta in angles:
        if j == 0:
            cudaq.control(y_rot, qubits, aux, theta)
        else:
            tot_x = num_x[j-1]
            for i in range(tot_x):
                x(qubits[index_q[count]])
                count += 1
            cudaq.control(y_rot, qubits, aux, theta)
        j += 1



###############################################################################################################################################
# Ansatz Implementations (NO qvector allocation here)
###############################################################################################################################################

from typing import List
import cudaq

@cudaq.kernel
def ansatz(
    q: cudaq.qview,
    n_class_qubits: int,
    n_map_qubits: int,
    attention_params: List[float],
    processing_params: List[float],
    num_layers: int = 2,
):
    """
    Nearest-neighbor ansatz using RY and controlled gates.
    """

    total_qubits = n_class_qubits + n_map_qubits

    a = 0
    map_start = n_class_qubits
    map_end   = n_class_qubits + n_map_qubits

    for k in range(map_start, map_end):
        if a < len(attention_params):
            ry(attention_params[a], q[k])
            a += 1

    if n_map_qubits >= 2:
        for k in range(map_start, map_end - 1):
            if a < len(attention_params):
                cry(attention_params[a], q[k], q[k + 1])
                a += 1

    p = 0
    L = num_layers if num_layers > 0 else 1

    class_start = 0
    class_end   = n_class_qubits

    for _ in range(L):

        for idx in range(total_qubits):
            if p < len(processing_params):
                ry(processing_params[p], q[idx])
                p += 1

        if n_class_qubits >= 2:
            for c in range(class_start, class_end - 1):
                x.ctrl(q[c], q[c + 1])

        if n_map_qubits >= 2:
            for k in range(map_start, map_end - 1):
                x.ctrl(q[k], q[k + 1])


###############################################################################################################################################
# Final Circuit/Kernel Constructor (single allocation point)
###############################################################################################################################################
@cudaq.kernel
def kernel_constructor(
    n_qubits: int,
    n_class_qubits: int,
    n_map_qubits: int,
    FEATURE_MAP: int,
    ANSATZ: int,
    input_params: List[float],
    index_q: List[int],
    num_x: List[int],
    pos_pixel: int,
    attention_params: List[float],
    processing_params: List[float],
    num_layers: int = 2,
):
    """
    Construct the full quantum circuit kernel with specified feature map and ansatz.
    """

    # Single allocation point
    q = cudaq.qvector(n_qubits)


    if FEATURE_MAP == 0:
        frqi(
            pos_pixel,
            input_params,
            index_q,
            num_x,
         )

    if ANSATZ == 0:
        ansatz(
            q,
            n_class_qubits,
            n_map_qubits,
            attention_params,
            processing_params,
            num_layers,
        )