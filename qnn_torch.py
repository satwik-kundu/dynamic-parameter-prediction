from sklearn.datasets import load_breast_cancer
import app.qnn_builder as qnn_builder
import pennylane as qml
from sklearn.preprocessing import normalize
from numpy import genfromtxt
import sys
import os
import qiskit, qiskit_aer
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from prediction import *
from train_function import *
import csv
import pdb
import time
import copy
from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt

g = 10
torch.manual_seed(g)
np.random.seed(g)

def func(x, m, c):
    return m * x + c


def get_plot(results, labels):
    fig = plt.figure(figsize=(8, 8))
    x = [i + 1 for i in range(epochs)]

    titles = ["Train_Accuracy", "Train_Loss", "Test_Accuracy", "Test_Loss"]

    for i in range(len(titles)):
        ax = fig.add_subplot(2, 2, i + 1)
        conv_rate = []
        speedup = []

        for j in range(len(results)):
            ax.plot(x, results[j][i], label=f"{labels[j]}")
            ax.legend()

            para, _ = curve_fit(func, x, results[j][i])
            m, c = para
            conv_rate.append(abs(m))

        ax.set_title(f"{titles[i]}")
        np.savetxt(
            out_prefix + f"/{titles[i]}_convrate.csv",
            np.array(conv_rate),
            delimiter=",",
        )
        
    fig.show()
    fig.savefig(out_prefix + f"/plots.png")


data = genfromtxt(sys.argv[1], delimiter=",")
X = data[:, :-1]
X = normalize(X, axis=0, norm="max")
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)

features = X.shape[-1]
classes = int(max(y) + 1)

# build qnn model
qubit = 4
layers = 1
enc = 2
pqc = 4
meas = 3

# Initialization
lr = 0.005
batch_size = 32
epochs = 100
p = 5
d = 3
k = 0.0001
dr = 0.95

out_prefix = sys.argv[1].split("/")[-1].split(".")[0] + f"_{qubit}q_{pqc}pqc"


# check compatibility between the chosen model and the dataset
qnn = qnn_builder.PennylaneQNNCircuit(
    enc=enc, qubit=qubit, layers=layers, pqc=pqc, meas=meas
)
input_length = qnn.enc_builder.max_inputs_length()
assert features <= input_length

# build the quantum-classical hybrid learning model
weight_shape = qnn.pqc_builder.weigths_shape()
if isinstance(weight_shape[0], tuple):
    ql_weights_shape = {"weights0": weight_shape[0], "weights1": weight_shape[1]}
else:
    ql_weights_shape = {"weights0": weight_shape, "weights1": ()}
output_dim = qnn.meas_builder.output_dim()


dev = qml.device("lightning.qubit", wires=qubit)

noisy_dev = qml.device("default.mixed", wires=qubit)
noisy_dev = qml.transforms.insert(qml.AmplitudeDamping, 0.1, position="all")(noisy_dev)
noisy_dev._rng = np.random.default_rng(g)

qnode = qml.QNode(qnn.construct_qnn_circuit, dev, interface="torch", diff_method="adjoint")
noisy_qnode = qml.QNode(qnn.construct_qnn_circuit, noisy_dev, interface="torch", diff_method="spsa")

qlayer = qml.qnn.TorchLayer(qnode, ql_weights_shape)
noisy_qlayer = qml.qnn.TorchLayer(noisy_qnode, ql_weights_shape)
clayer = torch.nn.Linear(output_dim, classes)
model = torch.nn.Sequential(qlayer, clayer)
noisy_model = torch.nn.Sequential(noisy_qlayer, clayer)

# PyTorch
loss_fn = torch.nn.CrossEntropyLoss()

# train dataset
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

num_param = int(sum([np.prod(param.shape) for param in model.parameters()]))
w_hist = torch.empty((epochs, num_param))

results = []
labels = []

labels.append("Noisy")
torch.save(model.state_dict(), 'model.pth')
results.append(
    prediction_train(
        None,
        epochs,
        w_hist,
        noisy_model,
        train_loader,
        x_train,
        y_train,
        x_test,
        y_test,
        lr,
        loss_fn,
        p,
        d,
        k,
        dr,
        out_prefix + "/noisy",
    )
)



model.load_state_dict(torch.load('model.pth'), strict=True)
results.append(
    prediction_train(
        naive_prediction,
        epochs,
        w_hist,
        model,
        train_loader,
        x_train,
        y_train,
        x_test,
        y_test,
        lr,
        loss_fn,
        p,
        d,
        k,
        dr,
        out_prefix + "/nap",
    )
)

num_param = int(sum([np.prod(param.shape) for param in noisy_model.parameters()]))
w_hist = torch.empty((epochs, num_param))
labels.append("Noiseless")
# model.load_state_dict(torch.load('model.pth'), strict=True)
results.append(
    prediction_train(
        None,
        epochs,
        w_hist,
        model,
        train_loader,
        x_train,
        y_train,
        x_test,
        y_test,
        lr,
        loss_fn,
        p,
        d,
        k,
        dr,
        out_prefix + "/noiseless",
    )
)


get_plot(results, labels)
