import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

st.title("CNN")

# Features
st.sidebar.header("Model Parameters")
num_neurons_layer1 = st.sidebar.slider("Number of neurons in Layer 1", min_value=1, max_value=50, value=10)
num_neurons_layer2 = st.sidebar.slider("Number of neurons in Layer 2", min_value=1, max_value=50, value=10)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=50, value=10)

mc_dropout_enabled = st.sidebar.checkbox("Enable MC Dropout", value=False)

dataset_choice = st.sidebar.selectbox("Dataset", ["Circle", "Gaussian", "Exclusive OR"])

basis_function = st.sidebar.selectbox("Basis Function", ["None", "Sine", "Gaussian"])

activation_function = st.sidebar.selectbox("Activation Function", ["Sigmoid", "ReLU", "Tanh"])

# model and training function
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_neurons_layer1, num_neurons_layer2, activation_fn):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, num_neurons_layer1)
        self.layer2 = nn.Linear(num_neurons_layer1, num_neurons_layer2)
        self.output_layer = nn.Linear(num_neurons_layer2, 1)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.layer1(x))
        x = self.activation_fn(self.layer2(x))
        return torch.sigmoid(self.output_layer(x))

def train_model(X_train, y_train, num_epochs, learning_rate, num_neurons_layer1, num_neurons_layer2, activation_fn):
    model = NeuralNetwork(input_size=X_train.shape[1], num_neurons_layer1=num_neurons_layer1, num_neurons_layer2=num_neurons_layer2, activation_fn=activation_fn)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    return model

# Load and preprocess the dataset
def load_circle_dataset():
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=100, noise=0.1, random_state=42, factor=0.5)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y.reshape(-1, 1))
    return X, y

def load_gaussian_dataset():
    from sklearn.datasets import make_gaussian_quantiles
    X, y = make_gaussian_quantiles(n_samples=100, n_features=2, n_classes=2, random_state=42)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y.reshape(-1, 1))
    return X, y

def load_exclusive_or_dataset():
    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.FloatTensor([[0], [1], [1], [0]])
    return X, y

def apply_basis_function(X, basis_function):
    if basis_function == "Sine":
        X_transformed = torch.sin(X)
    elif basis_function == "Gaussian":
        X_transformed = torch.exp(-X ** 2)
    else:
        X_transformed = X
    return X_transformed

if dataset_choice == "Circle":
    X_train, y_train = load_circle_dataset()
elif dataset_choice == "Gaussian":
    X_train, y_train = load_gaussian_dataset()
else:
    X_train, y_train = load_exclusive_or_dataset()

X_train = apply_basis_function(X_train, basis_function)

# Step 4: Define the contour plot function
def create_contour_plot(model, X, y, mc_dropout_enabled=False):
    # Generate a meshgrid over the input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Flatten the grid and make predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points = torch.FloatTensor(grid_points)
    grid_points = apply_basis_function(grid_points, basis_function)
    with torch.no_grad():
        if mc_dropout_enabled:
            model.train()  
        else:
            model.eval()   
        probabilities = model(grid_points).numpy()

    zz = probabilities.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, zz, cmap='RdYlBu', alpha=0.8)
    ax.scatter(X[y[:, 0] == 0][:, 0], X[y[:, 0] == 0][:, 1], c='red', edgecolors='none', label='Class 0')
    ax.scatter(X[y[:, 0] == 1][:, 0], X[y[:, 0] == 1][:, 1], c='yellow', edgecolors='none', label='Class 1')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Predictions")
    ax.legend()

    # Show the plot using st.pyplot() with the figure and axis objects
    st.pyplot(fig)


if __name__ == "__main__":
    if st.button("Train Model"):
        activation_fn = nn.Sigmoid() if activation_function == "Sigmoid" else nn.ReLU() if activation_function == "ReLU" else nn.Tanh()
        model = train_model(X_train, y_train, num_epochs, learning_rate, num_neurons_layer1, num_neurons_layer2, activation_fn)
        create_contour_plot(model, X_train, y_train, mc_dropout_enabled)
