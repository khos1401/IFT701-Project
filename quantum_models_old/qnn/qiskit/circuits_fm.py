from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes, PauliFeatureMap

class qiskitFeatureMaps:

    def __init__(self, num_features, num_qubits, num_reps=2):
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.num_reps = num_reps

    def create_ry_map(self):
        """
        Create a simple feature map using Qiskit. Simple RY encoding with minimal entanglement (CX).
        """

        feature_map = QuantumCircuit(self.num_features)

        k = min(self.num_features, self.num_qubits)

        for i in range(k):
            feature_map.ry(Parameter(f'Input_{i}'), i)

        for i in range(k - 1):
            feature_map.cx(i, i + 1)
        
        return feature_map
    
    def create_rx_ry_map(self):
        """
        Create a feature map using Qiskit. Slightly more complex encoding with RZ and RY gates and entanglement (CX).
        """

        feature_map = QuantumCircuit(self.num_features)

        for i in range(self.num_features):
            feature_map.ry(Parameter(f'Input_{i}'), i)
            feature_map.rz(Parameter(f'Input_{i}'), i)

        for i in range(self.num_qubits - 1):
            feature_map.cx(i, i + 1)

        return feature_map
    
    def create_zfeature_map(self):
        """
        Create a feature map using Qiskit SDK. ZFeatureMap with no entanglement.
        """

        feature_map = ZFeatureMap(
            feature_dimension=min(self.num_features, self.num_qubits), 
            reps=self.num_reps, 
            name='ZFeatureMap').decompose()

        return feature_map
    
    def create_zzfeature_map(self):
        """
        Create a feature map using Qiskit SDK. ZZFeatureMap with entanglement.
        """

        feature_map = ZZFeatureMap(
            feature_dimension=min(self.num_features, self.num_qubits), 
            reps=self.num_reps,
            entanglement='linear',
            name='ZZFeatureMap').decompose()

        return feature_map
    
    def create_pauli_feature_map(self):
        """
        Create a feature map using Qiskit SDK. PauliFeatureMap with entanglement.
        """

        feature_map = PauliFeatureMap(
            feature_dimension=min(self.num_features, self.num_qubits), 
            reps=self.num_reps,
            paulis=['Z', 'ZZ', 'X', 'XX'],
            entanglement='linear',
            name='PauliFeatureMap').decompose()
        
        return feature_map
    
    def create_real_amplitudes_map(self):
        """
        Create a feature map using Qiskit SDK. RealAmplitudes with entanglement.
        """

        feature_map = RealAmplitudes(
            feature_dimension=min(self.num_features, self.num_qubits), 
            reps=self.num_reps,
            entanglement='reverse_linear',
            name='RealAmplitudes').decompose()

        return feature_map