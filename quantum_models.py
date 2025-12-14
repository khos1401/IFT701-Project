
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


class QuantumNN(nn.Module):
    def __init__(self,
        input_size: list[int]
    ):
        self.input_size = input_size
        n_qubits = int(np.ceil(np.log2(np.prod(input_size))))
        n_layers = 4

        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch") #Add diff_method="parameter-shift" for more realistic but much slower
        def quantum_circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        super(QuantumNN, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.c_layer = nn.Linear(n_qubits, 2)
        for param in self.c_layer.parameters():
            param.requires_grad = False # Not training classical layer

        nn.init.xavier_uniform_(self.c_layer.weight)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.q_layer(x)
        x = self.c_layer(x)
        return x
    
    def get_model_info(self):
        """Return model architecture and total parameters."""
        return {
            'model_type': 'QuantumNN',
            'input_size': self.input_size,
            'hidden_dims': 4,
            'num_classes': 2,
            'dropout': 0,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': str(self)
        }


def quantum_convolution(control, target, weights):
    qml.U3(*weights[0:3], wires=control)
    qml.U3(*weights[3:6], wires=target)
    qml.CNOT(wires=[control, target])


def quantum_pooling(control, target, weights):
    qml.U3(*weights[0:3], wires=control)
    qml.U3(*weights[0:3], wires=target)
    qml.CRot(*weights[6:9], wires=[control, target])


def get_CNN_param_count(n_qubits):
    total_params = 0
    active_qubits = n_qubits
    while active_qubits > 1:
        total_params += 6 * (active_qubits//2 + (active_qubits-1)//2)
        n_pools = active_qubits // 2
        total_params += n_pools * 9
        active_qubits = active_qubits - n_pools
    return total_params

class QuantumCNN(nn.Module):
    def __init__(self,
        input_size: list[int]
    ):
        self.input_size = input_size
        n_qubits = int(np.ceil(np.log2(np.prod(input_size))))

        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch") #Add diff_method="parameter-shift" for more realistic but much slower
        def quantum_circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
            qubits_left = list(range(n_qubits))
            param_idx = 0
            while len(qubits_left) > 1:
                n_active = len(qubits_left)
                for i in range(0, n_active-1, 2):
                    quantum_convolution(qubits_left[i], qubits_left[(i+1)%n_active], weights[param_idx:param_idx+12])
                    param_idx += 6
                for i in range(1, n_active-1, 2):
                    quantum_convolution(qubits_left[i], qubits_left[(i+1)%n_active], weights[param_idx:param_idx+6])
                    param_idx += 6
                
                new_qubits_left = []
                for i in range(0, n_active-1, 2):
                    quantum_pooling(qubits_left[i], qubits_left[i+1], weights[param_idx:param_idx+9])
                    param_idx += 9
                    new_qubits_left.append(qubits_left[i+1])
                if n_active % 2 == 1:
                    new_qubits_left.append(qubits_left[-1])
                qubits_left = new_qubits_left
            return qml.probs(wires=qubits_left[0])

        super(QuantumCNN, self).__init__()
        weight_shapes = {"weights": (get_CNN_param_count(n_qubits),)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.q_layer(x)
        return x
    
    def get_model_info(self):
        """Return model architecture and total parameters."""
        return {
            'model_type': 'QuantumNN',
            'input_size': self.input_size,
            'hidden_dims': 4,
            'num_classes': 2,
            'dropout': 0,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'architecture': str(self)
        }
    
