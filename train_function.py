import torch
import os
from pennylane import numpy as np
import qiskit, qiskit_aer
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.fake_provider import FakeManila, FakeNairobi, FakeVigo, FakeLima
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error, errors
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(7)

def get_reduced_noise_model(scaling_factor):
    noise_model = qiskit_aer.noise.NoiseModel.from_backend(FakeNairobi()) 
    reduced_noise_model = NoiseModel(basis_gates=noise_model.basis_gates)

    # Iterate through each gate in the real noise model
    for gate, noise_qubits in noise_model._default_quantum_errors.items():
        
        for qubit, qerror in enumerate(noise_qubits):
            if qerror is None:
                continue
            
            print(qerror)
            # Get the error terms and probabilities
            error_terms = qerror.terms
            probabilities = qerror.probabilities

            # Scale down the probabilities
            probabilities = [p * scaling_factor for p in probabilities]

            # Create a new quantum error with the scaled probabilities
            new_qerror = errors.QuantumError(list(error_terms), probabilities)

            # Add the new quantum error to the new noise model
            reduced_noise_model.add_quantum_error(new_qerror, gate, [qubit])
    
    return reduced_noise_model

def custom_noise_model():
    # Error probabilities
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01   # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = depolarizing_error(prob_1, 1)
    error_2 = depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return noise_model

def save_file(outdir, tr_acc, tr_loss, test_acc, test_loss, weights):
    np.savetxt(outdir + "/train_loss.csv", np.array(tr_loss), delimiter=",")
    np.savetxt(outdir + "/train_acc.csv", np.array(tr_acc), delimiter=",")
    np.savetxt(outdir + "/test_loss.csv", np.array(test_loss), delimiter=",")
    np.savetxt(outdir + "/test_acc.csv", np.array(test_acc), delimiter=",")
    np.savetxt(outdir + "/weights.csv", np.array(weights), delimiter=",")


def prediction_train(
    prediction,
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
    outdir,
):
    os.makedirs(outdir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # collect statistics
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(epochs):
        start = time.time()

        w_temp = [param.data.view(-1) for param in model.parameters()]
        w_hist[epoch] = torch.cat(w_temp)

        if (epoch + 1) % p == 0 and prediction != None:
            start, end = 0, 0
            for param in model.parameters():
                start = end
                end = start + int(np.prod(param.shape))
                new_param = prediction(
                    p - 1,
                    w_hist[(epoch - p + 2) : (epoch + 1), start:end],
                    d,
                    k,
                    lr,
                    dr,
                    epoch,
                )
                param.data.copy_(new_param.reshape(param.shape))

        else:
            for x_batch, y_batch in train_loader:
                # forward pass
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                opt.zero_grad()
                loss.backward()

                # update weights
                opt.step()

        with torch.no_grad():
            # evaluate and store train metrics
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            acc = (torch.argmax(y_pred, dim=1) == y_train).float().mean()
            train_acc.append(acc)
            train_loss.append(loss)

            # evaluate and store test metrics
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            acc = (torch.argmax(y_pred, dim=1) == y_test).float().mean()
            test_acc.append(acc)
            test_loss.append(loss)

        end = time.time()
        # print progress
        print(
            f"\nEpoch {epoch + 1}: Time: {(end - start):.2f}s, Train loss: {train_loss[-1]:.4f}, Train acc: {train_acc[-1]*100:.2f}%, Test loss: {test_loss[-1]:.4f}, Test acc: {test_acc[-1]*100:.2f}%"
        )
    save_file(outdir, train_acc, train_loss, test_acc, test_loss, w_hist)
    return [train_acc, train_loss, test_acc, test_loss, w_hist]

def split_and_tensorize(X, y, test_size=0.3, shuffle=True):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, y_train, x_test, y_test

def plot_results(results, labels, out_prefix, epochs):
    fig = plt.figure(figsize=(8, 8))
    x = [i + 1 for i in range(epochs)]

    titles = ["Train_Accuracy", "Train_Loss", "Test_Accuracy", "Test_Loss"]

    for i in range(len(titles)):
        ax = fig.add_subplot(2, 2, i + 1)

        for j in range(len(results)):
            ax.plot(x, results[j][i], label=f"{labels[j]}")
            ax.legend()

        ax.set_title(f"{titles[i]}")

    fig.show()
    fig.savefig(out_prefix + f"/plots_{epochs}e.png")