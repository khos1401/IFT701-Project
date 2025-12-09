import sys
import os
import numpy as np
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
)

import torch
import torch.nn as nn

from quantum_models.qnn.cudaq.circuits import (
    QuantumCircuitConfig,
    FeatureMapType,
    AnsatzType,
    kernel_constructor,
)

import cudaq


def setup_cudaq():
    """
    Sets up the CUDA-Q environment for quantum circuit execution.
    """
    print("Setting up CUDA-Q environment...")
    cudaq.set_target("nvidia")
    backend = "nvidia"
    return backend


class QuantumNeuralNetwork(nn.Module):
    """
    Quantum neural network wrapper around a CUDA-Q kernel.
    """

    def __init__(
        self,
        config: QuantumCircuitConfig,
        num_classes: int = 2,
        use_classical_head: bool = True,
    ):
        super().__init__()
        self.config = config

        # Enforce binary classification
        assert num_classes == 2
        self.num_classes = num_classes
        self.use_classical_head = use_classical_head

        # Learnable quantum circuit parameters
        self.attention_params = nn.Parameter(
            0.1 * torch.randn(config.attention_params)
        )
        self.processing_params = nn.Parameter(
            0.1 * torch.randn(config.processing_params)
        )

        # Optional classical head on top of quantum probabilities
        if use_classical_head:
            self.classifier = nn.Sequential(
                nn.Linear(self.num_classes, 16, bias=True),
                nn.ReLU(),
                nn.Linear(16, self.num_classes, bias=True),
            )
        else:
            # Identity: directly return quantum probabilities as outputs
            self.classifier = nn.Identity()

    @staticmethod
    def mcqi_rgb(image):
        """
        Convert an RGB image into MCQI angle encoding:
            image → [[θ_R, θ_G, θ_B], ...] + FRQI-style index tracking.
        """

        flat = image.reshape(-1, 3)

        if flat.max() > 1.0:
            flat = flat / 255.0

        angles_rgb = []
        for (r, g, b) in flat:
            θr = np.arcsin(r) if r != 0 else 0.0
            θg = np.arcsin(g) if g != 0 else 0.0
            θb = np.arcsin(b) if b != 0 else 0.0
            angles_rgb.append([θr, θg, θb])

        pos_pixel = int(np.log2(len(angles_rgb)))
        k_value = pos_pixel - 1

        def change(state, new_state):
            n = len(state)
            c = np.array([])
            for i in range(n):
                if state[i] != new_state[i]:
                    c = np.append(c, int(i))
            return c.astype(int) if len(c) > 0 else c

        index = []

        for j in range(len(angles_rgb)):
            state    = format(j - 1, f"0{pos_pixel}b")
            newstate = format(j, f"0{pos_pixel}b")

            if j != 0:
                c = change(state, newstate)
                if len(c) > 0:
                    temp = np.abs(c - k_value)
                    index.append(temp)

        index_q = []
        num_x = []

        for arr in index:
            arr = arr.tolist()
            num_x.append(len(arr))
            index_q.extend(arr)

        return angles_rgb, index_q, num_x, pos_pixel
    
    @staticmethod
    def extract_bit(
        bitstring: str,
        qubit_indices,
        lsb_right: bool = True,
    ) -> str:
        """
        Extracts a new bitstring containing only the bits at the specified qubit indices.
        """
        n = len(bitstring)
        bits = []
        for q in sorted(qubit_indices):  # ensure consistent order
            pos = (n - 1 - q) if lsb_right else q
            bits.append(bitstring[pos])
        return "".join(bits)
    
    def counts_to_half_split_binary(
        self,
        counts: dict,
        n_qubits: int,
    ) -> torch.Tensor:
        """
        Map full-register measurement counts into a binary distribution
        by splitting the 2^n computational basis states into two halves.
        """
        num_classes = 2
        buckets = [0, 0]
        total = 0

        half = 1 << (n_qubits - 1)   # 2^{n-1}

        for bitstring, c in counts.items():
            idx = int(bitstring, 2)

            if idx < half:
                cls = 0
            else:
                cls = 1

            buckets[cls] += c
            total += c

        if total == 0:
            return torch.zeros(num_classes, dtype=torch.float32)

        p0 = buckets[0] / total
        p1 = buckets[1] / total
        return torch.tensor([p0, p1], dtype=torch.float32)


    def quantum_forward(
        self,
        input_batch: torch.Tensor,
        shots: int = 1024,
    ) -> torch.Tensor:
        """
        Runs the quantum circuit for each RGB image in the batch 
        and returns quantum output probabilities for binary classification.
        """

        device = input_batch.device
        batch_size = input_batch.shape[0]

        kernel = kernel_constructor

        # Model config
        n_qubits = self.config.num_qubits
        n_class_qubits = self.config.n_class_qubits
        n_map_qubits = self.config.n_map_qubits
        num_layers = self.config.num_reps

        # Map enums for CUDA-Q
        fm_map = {FeatureMapType.BASIC_RY: 0}
        an_map = {AnsatzType.RING: 0}

        feature_map = fm_map[self.config.feature_map_type]
        ansatz = an_map[self.config.ansatz_type]

        # Parameters (torch → python list)
        att = self.attention_params.detach().cpu().tolist()
        proc = self.processing_params.detach().cpu().tolist()

        quantum_outputs = []

        for i in range(batch_size):

            image_np = input_batch[i].detach().cpu().numpy()

            angles_rgb, index_q, num_x, pos_pixel = self.mcqi_rgb(image_np)

            flat_angles = [angle for triplet in angles_rgb for angle in triplet]

            counts = cudaq.sample(
                kernel,
                n_qubits,
                n_class_qubits,
                n_map_qubits,
                feature_map,
                ansatz,

                # feature map
                flat_angles,      
                index_q,         
                num_x,           
                pos_pixel,         

                # ansatz
                att,
                proc,
                num_layers,
                shots_count=shots,
            )

            probs = self.counts_to_half_split_binary(
                counts=counts,
                n_qubits=n_qubits,
            ).to(device)

            quantum_outputs.append(probs)

        # Batch output [B, 2]
        return torch.stack(quantum_outputs, dim=0)

    def forward(self, x: torch.Tensor, shots: int = 1024) -> torch.Tensor:

        quantum_probs = self.quantum_forward(x, shots=shots)
        outputs = self.classifier(quantum_probs)
        return outputs
