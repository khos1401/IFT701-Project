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


####################################################################################################################################################


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
            0.1 * torch.randn(config.num_attention_params)
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

    @staticmethod # https://arxiv.org/pdf/2404.06889
    def frqi(image):

        faltten_image = image.flatten()
        angles=[]
        for intensity in faltten_image:
            if intensity == 0:
                angles.append(0.0)
            else:
                theta = np.arcsin(intensity)
                angles.append(theta)

        # number of qubits
        pos_pixel = int(np.log2(len(angles)))

        k_value = pos_pixel-1

        # This function let us know which are the qubits that need to be applied
        # the X-gate so we can change the state of the pixels positions qubits
        # to the new state.

        def change(state, new_state):

            n = len(state)  # n is the length of the binary string
            c = np.array([])  # create an empty array
            for i in range(n):  # start to iterate n times
                if state[i] != new_state[i]:
                    c = np.append(c, int(i))  # if it is different we append the position to the array

            if len(c) > 0:
                return c.astype(int)
            else:
                return c

        index=[]

        for jk in range(len(angles)):
            state = '{0:0{1}b}'.format(jk-1, pos_pixel)
            new_state = '{0:0{1}b}'.format(jk, pos_pixel)

            if jk != 0:
                c = change(state, new_state)
                if len(c) > 0:
                    temp = np.abs(c-k_value)
                    index.append(temp)

        index_q = []
        num_x = []

        for arr in index:
            count = 0
            arr = arr.tolist()
            for idx in arr:
                index_q.append(idx)
                count += 1
            num_x.append(count)

        return angles, index_q, num_x, pos_pixel
        
    @staticmethod
    def counts_to_class_probs(counts: dict, total_qubits: int) -> torch.Tensor:
        p0 = 0
        p1 = 0
        total = 0

        for bitstring, c in counts.items():
            # Right-most bit (aux) is the classifier bit
            cls_bit = bitstring[-1]
            if cls_bit == '0':
                p0 += c
            else:
                p1 += c
            total += c

        if total == 0:
            return torch.tensor([0.5, 0.5], dtype=torch.float32)

        return torch.tensor([p0/total, p1/total], dtype=torch.float32)


    def quantum_forward(
        self,
        input_batch: torch.Tensor,
        shots: int = 1024,
    ) -> torch.Tensor:
        """
        Runs the quantum circuit for each image in the batch 
        and returns quantum output probabilities for binary classification.
        """

        device = input_batch.device
        batch_size = input_batch.shape[0]

        kernel = kernel_constructor

        # Model config
        n_qubits = self.config.n_qubits
        n_qubit_total = self.config.n_qubits + 1  # +1 for auxiliary qubit
        num_layers = self.config.num_reps

        feature_map = 0
        ansatz = 0

        # Parameters (torch → python list)
        att = self.attention_params.detach().cpu().tolist()

        quantum_outputs = []

        for i in range(batch_size):

            image_np = input_batch[i].detach().cpu().numpy()

            angles, index_q, num_x, pos_pixel = self.frqi(image_np)

            n_qubit_total = pos_pixel + 1  # +1 for auxiliary qubit

            counts = cudaq.sample(
                kernel,
                pos_pixel,
                feature_map,
                ansatz,
                angles,      
                index_q,         
                num_x,                 
                att,
                num_layers,
                shots_count=shots,
            )

            probs = self.counts_to_class_probs(
                counts=counts,
                total_qubits=n_qubit_total,
            ).to(device)

            quantum_outputs.append(probs)

        # Batch output [B, 2]
        return torch.stack(quantum_outputs, dim=0)

    def forward(self, x: torch.Tensor, shots: int = 1024) -> torch.Tensor:

        quantum_probs = self.quantum_forward(x, shots=shots)
        outputs = self.classifier(quantum_probs)
        return outputs
