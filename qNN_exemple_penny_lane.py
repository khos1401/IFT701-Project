from utils import get_data_tensors, evaluate_model, save_results, save_training_history, plot_training_history, compute_multiclass_roc, plot_multiclass_roc
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim


dataset_path = 'dataset/mnist_8x8.npz'
class_to_keep = [0, 1]
batch_size = 32


X_train, X_test, X_val, y_train, y_test, y_val, class_names = get_data_tensors(dataset_path, class_to_keep)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

print(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples, {X_test.shape[0]} test samples.")




n_qubits = 6  # 2^6 = 8x8
n_layers = 4 
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class AmplitudeHybridQNN(nn.Module):
    def __init__(self):
        super(AmplitudeHybridQNN, self).__init__()
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.cl_layer_out = nn.Linear(n_qubits, 2)
        for param in self.cl_layer_out.parameters():
            param.requires_grad = False
        
        nn.init.xavier_uniform_(self.cl_layer_out.weight)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.q_layer(x)
        x = self.cl_layer_out(x)
        return x



model = AmplitudeHybridQNN()
optimizer = optim.Adam(model.parameters(), lr=0.005) 
criterion = nn.CrossEntropyLoss()
epochs = 10
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}


for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(output, 1)
        train_correct += torch.sum(preds == y_batch.data)

    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(output, 1)
            val_correct += torch.sum(preds == y_batch.data)


    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct.double() / len(val_loader.dataset)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc.item())
    history['val_acc'].append(val_acc.item())

    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')



#save_training_history(history, 'history_amplitude.json')
#plot_training_history(history)
#evaluate_model(model, test_loader, class_names)



