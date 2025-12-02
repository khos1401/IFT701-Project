import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '../../../..'))))

import torch
import torch.nn as nn

from quantum_models.qnn.cudaq.circuits import (
    QuantumCircuitConfig, FeatureMapType, AnsatzType, kernel_constructor
    )

import cudaq
import cuquantum


def setup_cudaq():
    """
    Sets up the CUDA-Q environment for quantum circuit execution.
    """

    print('Setting up CUDA-Q environment...')
    cudaq.set_target('nvidia')
    backend = 'nvidia'
        
    return backend

class QuantumNeuralNetwork(nn.Module):

    def __init__(
                self, 
                config: QuantumCircuitConfig,
                ):
        """
        Initialize the QuantumNeuralNetwork.
        """

        super().__init__()
        self.config = config

        # Learnable quantum circuit parameters
        self.attention_params = nn.Parameter(torch.randn(config.attention_params) * 0.1)
        self.processing_params = nn.Parameter(torch.randn(config.processing_params) * 0.1)

        # Classical classifier (linear layer)
        self.classifier = nn.Sequential(
            nn.Linear(config.n_class_qubits, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, config.n_class_qubits, bias=True),
        )


    def extract_bit(self, bitstring: str, qubit_indices, lsb_right: bool = True) -> str:
        """
        Extracts a new bitstring containing only the bits at the specified qubit indices.
        """

        n = len(bitstring)
        bits = []
        for q in sorted(qubit_indices): # ensure consistent order
            pos = (n - 1 - q) if lsb_right else q
            bits.append(bitstring[pos])
        return "".join(bits)

    def counts_to_class_probs(self, counts: dict,
                              num_classes: int,
                              class_qubits: int,
                              lsb_right: bool = True) -> torch.Tensor:
        """
        Sum measurement counts into a length=num_classes probability vector.
        Uses binary codes on 'class_qubits' qubits to represent classes, then clips to num_classes.
        """

        assert class_qubits >= 1 and (1 << class_qubits) >= num_classes
        class_indices = list(range(class_qubits))
        buckets = [0] * num_classes
        total = 0

        for bitstring, c in counts.items():
            readout = self.extract_bit(bitstring, class_indices, lsb_right)
            cls = int(readout, 2)
            if cls < num_classes:
                buckets[cls] += c
            total += c

        if total == 0:
            return torch.zeros(num_classes, dtype=torch.float32)
        return torch.tensor([b / total for b in buckets], dtype=torch.float32)

    def quantum_forward(self, input_batch: torch.Tensor, shots: int = 1024) -> torch.Tensor:
        """
        Runs the quantum circuit for each input in the batch and returns quantum output probabilities.
        """

        batch_size = input_batch.shape[0]
        quantum_outputs = []

        kernel = kernel_constructor

        n_qubits = self.config.num_qubits
        n_class_qubits = self.config.n_class_qubits
        n_map_qubits = self.config.n_map_qubits
        num_layers = self.config.num_reps
        feature_map = self.config.feature_map_type
        ansatz = self.config.ansatz_type

        att = self.attention_params.tolist()
        proc = self.processing_params.tolist()

        fm_map = {
            FeatureMapType.BASIC_RY: 0,
        }

        an_map = {
            AnsatzType.RING: 0,
        }

        feature_map = fm_map.get(self.config.feature_map_type)
        ansatz = an_map.get(self.config.ansatz_type)
        assert feature_map is not None, f"Unknown feature map: {self.config.feature_map_type}"
        assert ansatz is not None, f"Unknown ansatz: {self.config.ansatz_type}"

        for i in range(batch_size):
            input_data = input_batch[i].float().tolist()
            counts = cudaq.sample(
                kernel,
                n_qubits, n_class_qubits, n_map_qubits,
                feature_map, ansatz,
                input_data, att, proc, num_layers,
                shots_count=shots
            )
            probs = self.counts_to_class_probs(
                counts,
                num_classes=self.config.n_class_qubits,
                class_qubits=self.config.n_class_qubits,
                lsb_right=True,
            ).to(
                device=input_batch.device,
            )

            quantum_outputs.append(probs)

        return torch.stack(quantum_outputs)
    
    def forward(self, x: torch.Tensor, shots: int = 1024) -> torch.Tensor:
        """
        Standard PyTorch forward pass: quantum results + classical classifier.
        """
        quantum_probs = self.quantum_forward(x, shots)
        logits = self.classifier(quantum_probs)
        return logits