import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
import torch.nn as nn
import torch

# Our libraries
from quantum_models.qnn.qiskit.circuits_ans import QuantumAnsatz
from quantum_models.qnn.qiskit.circuits_fm import qiskitFeatureMaps

def setup_gpu_sim(device='GPU', shots=64, method='statevector'):
    """
    Sets up the GPU simulator for Qiskit.
    
    Parameters
    ----------
    device : str, optional
        The device to use for simulation ('GPU' or 'CPU', default is 'GPU').
    shots : int, optional
        The number of shots for the simulation (default is 64).
    method : str, optional
        The simulation method to use ('statevector', 'unitary', etc., default is 'statevector').
    
    Returns
    -------
    simulator : AerSimulator
        The configured Qiskit Aer simulator.
    """

    available_devices = AerSimulator().available_devices()
    print(f"Available devices: {available_devices}")   

    # Highly optimized simulator options for speed
    simulator_options = {
        'method': 'statevector',  # Better GPU utilization than statevector /density_matrix
        'shots': shots,
        'seed_simulator': 42,
        'max_parallel_threads': 36,
        'max_parallel_experiments': 16,  # Increased for better GPU batch processing
        'max_parallel_shots': 16,        # Increased for better parallelization
        'num_threads_per_device': 42,    # Use all cores
        'statevector_parallel_threshold': 32,
        'statevector_sample_measure_opt': 32,
        'accept_distributed_results': True,
        'batched_shots_gpu': True,
        'batched_shots_gpu_max_qubits': 15,  # Increased to use more GPU memory
    }
    
    if device == 'GPU' and 'GPU' in available_devices:
        print("GPU")
        simulator_options.update({
            'device': 'GPU',
            'cuStateVec_enable': True,
            'max_memory_mb': 0,  # Use almost all 48GB VRAM
            'tensor_network_num_sampling_qubits': 14,
            'use_cuTensorNet_autotuning': True,
            'blocking_enable': False,
            'blocking_qubits': 10,
            'fusion_enable': True,
            'fusion_verbose': False,
            'fusion_max_qubit': 8,
            'fusion_threshold': 14,
            'extended_stabilizer_metropolis_mixing_time': 10,
        })        
        print("GPU acceleration enabled with maximum utilization settings")
        print(f"Configured for {shots} shots with density matrix method")
    else:
        simulator_options['device'] = 'CPU'
        print("Using CPU simulation")
        
    simulator = AerSimulator(**simulator_options)

    return simulator

class QCNN(nn.Module):
    """
    Implements a Quantum Convolutional Neural Network (QCNN) architecture using Qiskit.
    """

    def __init__(self, input_size: int, n_encoding_qubits: int, feature_map_type: str, n_class_qubits: int, use_gpu: bool = True, shots: int = 128):
        super(QCNN, self).__init__()
        self.input_size = input_size
        self.feature_map_type = feature_map_type
        self.n_encoding_qubits = n_encoding_qubits
        self.n_class_qubits = n_class_qubits
        self.n_qubits = n_encoding_qubits + n_class_qubits
        self.use_gpu = use_gpu
        self.shots = shots

        assert self.n_qubits == self.n_encoding_qubits + self.n_class_qubits, \
            f"Got n_qubits={self.n_qubits} but {self.n_encoding_qubits}+{self.n_class_qubits} != n_qubits"


        # Initialize feature maps and ansatz instances
        feature_maps_inst = qiskitFeatureMaps(num_features=input_size, num_qubits=self.n_qubits, num_reps=2)
        self.fm = self._select_feature_map(feature_maps_inst)
        ansatz_inst = QuantumAnsatz(input_size=input_size, n_encoding_qubits=n_encoding_qubits, n_class_qubits=n_class_qubits)
        self.ansatz = ansatz_inst.create_ansatz()

        # Create the full circuit
        self.qc = self.create_full_circuit()

        self.simulator = setup_gpu_sim(
            device = 'GPU' if use_gpu else 'CPU',
            shots = shots,
            method= 'statevector'
        )

        self.sampler = BackendSampler(backend=self.simulator)

        self.qcnn = SamplerQNN(
            circuit = self.qc,
            input_params=self.fm.parameters,
            weight_params=self.ansatz.parameters,
            sampler=self.sampler,
            output_shape=None, # Automatically determined
        )

        self.qlayer = TorchConnector(self.qcnn)

        # The output size will be 2^(total measured qubits)
        # We only measure the classification qubits
        quantum_output_size = 2 ** self.n_class_qubits

        # Final classical layer for interpretation of results
        self.output_layer = nn.Linear(quantum_output_size, self.n_class_qubits, bias=False)
        
        print(f"Optimized Quantum CNN initialized:")
        print(f"  - Total qubits: {self.n_qubits}")
        print(f"  - Encoding qubits: {self.n_encoding_qubits}")
        print(f"  - Classification qubits: {self.n_class_qubits}")
        print(f"  - Shots: {shots}")
        print(f"  - Quantum output size: {quantum_output_size}")
        print(f"  - Feature map type: {self.feature_map_type}")

    def _select_feature_map(self, feature_maps_inst: qiskitFeatureMaps):
        if self.feature_map_type == "ry":
            return feature_maps_inst.create_ry_map()
        elif self.feature_map_type == "rx_ry":
            return feature_maps_inst.create_rx_ry_map()
        elif self.feature_map_type == "z":
            return feature_maps_inst.create_zfeature_map()
        elif self.feature_map_type == "zz":
            return feature_maps_inst.create_zzfeature_map()
        elif self.feature_map_type == "pauli":
            return feature_maps_inst.create_pauli_feature_map()
        elif self.feature_map_type == "real_amplitudes":
            return feature_maps_inst.create_real_amplitudes_map()
        else:
            raise ValueError(f"Unknown feature_map_type: {self.feature_map_type}")

    def create_full_circuit(self) -> QuantumCircuit:
        """
        Constructs the full QCNN circuit by combining feature maps and ansatz layers.
        
        Returns
        -------
        QuantumCircuit
            The complete QCNN circuit.
        """
        qc = QuantumCircuit(self.n_qubits, self.n_class_qubits) # Specify classical register size
        qc.compose(self.fm, inplace=True)
        qc.compose(self.ansatz, inplace=True)

        for idx in range(self.n_class_qubits):
            qubit_idx = self.n_qubits - self.n_class_qubits + idx  # last block of qubits
            qc.measure(qubit_idx, idx)
        
        return qc
    

    def forward(self, x):
        """Alternative: Matrix multiplication method (fastest for many classes)"""
        quantum_output = self.qlayer(x)
        
        # Create mapping matrix once and reuse
        if not hasattr(self, 'mapping_matrix'):
            self.mapping_matrix = self._create_mapping_matrix(quantum_output.device)
        
        # Single matrix multiplication: [batch_size, n_states] @ [n_states, n_classes]
        class_probs = quantum_output @ self.mapping_matrix
        
        # Normalize
        class_probs = class_probs / (class_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        return class_probs

    def _create_mapping_matrix(self, device):
        """Create binary mapping matrix from quantum states to classes"""
        n_states = 2**self.n_qubits
        mapping_matrix = torch.zeros(n_states, 5, device=device)
        
        for i in range(n_states):
            bit_string = format(i, f'0{self.n_qubits}b')[-4:]
            
            if bit_string == '0000':
                mapping_matrix[i, 0] = 1
            elif bit_string == '0001':
                mapping_matrix[i, 1] = 1
            elif bit_string == '0010':
                mapping_matrix[i, 2] = 1
            elif bit_string == '0100':
                mapping_matrix[i, 3] = 1
            elif bit_string == '1000':
                mapping_matrix[i, 4] = 1
        
        return mapping_matrix