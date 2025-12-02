from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

class QuantumAnsatz:
    """
    Constructs a quantum ansatz (variational circuit) layer for quantum neural networks using Qiskit. 
    Convolutional and Pooling Layers are implemented to encode classical data into quantum states and
    provide a structured approach to quantum feature extraction. 
    """

    def __init__(self, input_size, n_encoding_qubits, n_class_qubits):
        self.input_size = input_size
        self.n_encoding_qubits = n_encoding_qubits
        self.n_class_qubits = n_class_qubits
        self.n_qubits = n_encoding_qubits + n_class_qubits

    def create_ansatz(self):
        ansatz = QuantumCircuit(self.n_qubits)

        assert self.n_qubits == self.n_encoding_qubits + self.n_class_qubits, \
            f"n_qubits={self.n_qubits} != {self.n_encoding_qubits}+{self.n_class_qubits}"

        # First Convolutional Layer
        conv1 = QuantumConvolutionalLayer(self.n_encoding_qubits, kernel_size=3, stride=1, name="conv1")
        conv1_circ = conv1.create_conv_circuit(num_params=conv1.kernel_size * conv1.n_qubits)
        ansatz.compose(conv1_circ, inplace=True)

        # Second Convolutional Layer
        conv2 = QuantumConvolutionalLayer(self.n_encoding_qubits, kernel_size=3, stride=1, name="conv2")
        conv2_circ = conv2.create_conv_circuit(num_params=conv2.kernel_size * conv2.n_qubits)
        ansatz.compose(conv2_circ, inplace=True)

        # Simple Pooling | ENCODING QUBITS --> CLASSIFICATION QUBITS
        E = self.n_encoding_qubits
        C = self.n_class_qubits
        base, rem = divmod(E, C)

        for ci in range(C):
            start = ci * base + min(ci, rem)
            end = start + base + (1 if ci < rem else 0)

            target = self.n_encoding_qubits + ci

            for j in range(start, end):
                ansatz.cx(j, target)                  
                ansatz.ry(Parameter(f'pool_{ci}_to_{j}'), target) 

        return ansatz

class QuantumConvolutionalLayer:
    """
    Constructs a quantum convolutional layer for quantum neural networks using Qiskit.
    """

    def __init__(self, n_qubits, kernel_size=2, stride=1, name="conv"):
        self.n_qubits = n_qubits
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name
        self.params = []

    def create_conv_circuit(self, num_params):
        qc = QuantumCircuit(self.n_qubits)
        
        params = [Parameter(f'{self.name}_param_{i}') for i in range(num_params)]
        self.params.extend(params)
        
        param_idx = 0
        # Single pass through the qubits
        for i in range(0, self.n_qubits - self.kernel_size + 1, self.stride):
            for j in range(self.kernel_size):
                if param_idx < len(params):
                    qc.ry(params[param_idx], i + j)
                    param_idx += 1
            
            # Single entangling gate per kernel
            if i + self.kernel_size - 1 < self.n_qubits:
                qc.cx(i, i + self.kernel_size - 1)
        
        return qc
