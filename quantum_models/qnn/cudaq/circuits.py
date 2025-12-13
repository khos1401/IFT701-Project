import cuquantum
import cudaq
import math
from typing import List
from enum import Enum


class FeatureMapType(Enum):
    FRQI = 'frqi'

class AnsatzType(Enum):
    ANSATZ = 'ansatz'


class QuantumCircuitConfig:
    def __init__(
        self,
        n_qubits: int,
        num_reps: int = 2,
        feature_map_type: FeatureMapType = FeatureMapType.FRQI,
        ansatz_type: AnsatzType = AnsatzType.ANSATZ,
        num_attention_params: int = 225,
    ):
        self.n_qubits = n_qubits
        self.num_reps = num_reps

        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type

        self.num_attention_params = num_attention_params

        self.total_params = num_attention_params * num_reps




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
def frqi(
    q: cudaq.qview, 
    aux: cudaq.qubit,
    angles: List[float], 
    index_q: List[int], 
    num_x: List[int]
):
    
    h(q)   # Hadamard on all position qubits

    j = 0
    count = 0
    for theta in angles:
        if j == 0:
            cudaq.control(y_rot, q, aux, theta)
        else:
            flips = num_x[j-1]
            for _ in range(flips):
                x(q[index_q[count]])
                count += 1
            cudaq.control(y_rot, q, aux, theta)
        j += 1



###############################################################################################################################################
# Ansatz Implementations (NO qvector allocation here)
###############################################################################################################################################


@cudaq.kernel
def ansatz(
    q: cudaq.qview,
    aux: cudaq.qubit,
    params: list[float],   # All params flattened: (2*(n+1))*num_layers
    num_layers: int
):
    n = len(q)
    p = 0  # param index

    for _ in range(num_layers):

        # RY + RZ on each data qubit
        for i in range(n):
            ry(params[p], q[i]); p += 1
            rz(params[p], q[i]); p += 1

        # RY + RZ on aux
        ry(params[p], aux); p += 1
        rz(params[p], aux); p += 1

        # q[0] → q[1] → ... → q[n-1]
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])

        # q[n-1] → aux
        x.ctrl(q[n - 1], aux)

        # aux → q[0] (close the ring)
        x.ctrl(aux, q[0])


###############################################################################################################################################
# Final Circuit/Kernel Constructor (single allocation point)
###############################################################################################################################################
@cudaq.kernel
def kernel_constructor(
    n_qubits: int,
    FEATURE_MAP: int,
    ANSATZ: int,
    input_params: List[float],
    index_q: List[int],
    num_x: List[int],
    attention_params: List[float],
    num_layers: int = 2,
):
    q = cudaq.qvector(n_qubits)
    aux = cudaq.qubit()

    if FEATURE_MAP == 0:
        frqi(
            q,
            aux,
            input_params,
            index_q,
            num_x,
        )

    if ANSATZ == 0:
        ansatz(
            q,
            aux,
            attention_params,
            num_layers,
        )