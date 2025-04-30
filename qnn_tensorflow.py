from sklearn.datasets import load_breast_cancer
import app.qnn_builder as qnn_builder
import pennylane as qml
import tensorflow as tf
from sklearn.preprocessing import normalize
from numpy import genfromtxt
from prediction import *
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
np.random.seed(7)

data = genfromtxt(sys.argv[1], delimiter=',')
X = data[:, :-1]
X = normalize(X, axis=0, norm='max')
y = data[:, -1]
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, shuffle=True)
features = X.shape[-1]
classes = max(y) + 1

# build qnn model
qubit = 4
layers = 1
enc = 2
pqc = 14
meas = 3
qnn = qnn_builder.PennylaneQNNCircuit(
    enc=enc, qubit=qubit, layers=layers, pqc=pqc, meas=meas)

original_epochs = 50
sample_epochs = 5
predict_epoch = 9
pr = 0.95  # prediction rate

out_prefix = 'qnn_results/' + \
    sys.argv[1].split('/')[-1].split('.')[0]
traindir = out_prefix + \
    f'/train_{original_epochs}o_{sample_epochs}s_{predict_epoch}p_{pr}pr_{qubit}q_{enc}e_{pqc}pqc'
os.makedirs(traindir, exist_ok=True)

# check compatibility between the chosen model and the dataset
input_length = qnn.enc_builder.max_inputs_length()
assert features <= input_length

# build the quantum-classical hybrid learning model
weight_shape = qnn.pqc_builder.weigths_shape()
if isinstance(weight_shape[0], tuple):
    ql_weights_shape = {
        'weights0': weight_shape[0], 'weights1': weight_shape[1]}
else:
    ql_weights_shape = {'weights0': weight_shape, 'weights1': ()}
output_dim = qnn.meas_builder.output_dim()

dev = qml.device("default.qubit", wires=qubit)  # target pennylane device
qnode = qml.QNode(qnn.construct_qnn_circuit, dev,
                  interface='tf', diff_method='backprop')  # circuit

qlayer = qml.qnn.KerasLayer(qnode, ql_weights_shape, output_dim=output_dim)
clayer = tf.keras.layers.Dense(classes)
model = tf.keras.models.Sequential([qlayer, clayer])


# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(opt, tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])

# store model weights after each epoch in original_weights dictionary
original_weights = {}
weight_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch,
    logs: original_weights.update({epoch: model.get_weights()}))

# model train
history_callback = model.fit(
    x_train, y_train, epochs=original_epochs, shuffle=True, batch_size=32, validation_data=(x_val, y_val), use_multiprocessing=False, callbacks=weight_callback)

tr_loss_history = history_callback.history['loss']
tr_acc_history = history_callback.history['accuracy']
val_loss_history = history_callback.history['val_loss']
val_acc_history = history_callback.history['val_accuracy']
np.savetxt(traindir + '/tr_loss_history.csv',
           np.array(tr_loss_history), delimiter=",")
np.savetxt(traindir + '/tr_acc_history.csv',
           np.array(tr_acc_history), delimiter=",")
np.savetxt(traindir + '/val_loss_history.csv',
           np.array(val_loss_history), delimiter=",")
np.savetxt(traindir + '/val_acc_history.csv',
           np.array(val_acc_history), delimiter=",")

# save model weights in a csv file
with open(traindir + '/original_weights.csv', 'w') as f:
    writer = csv.writer(f)
    for epoch, weights in original_weights.items():
        writer.writerow([epoch, weights])

tr_loss = []
tr_acc = []
val_loss = []
val_acc = []
var_list = []

while predict_epoch <= 9:

    # set model weights to predicted model weights after epochs=sample_epochs
    model.set_weights(naive_prediction(
        original_weights, sample_epochs, predict_epoch))

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(opt, tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    # store model weights after each epoch in sample_weights dictionary
    sample_weights = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch,
        logs: sample_weights.update({((
            epoch) % sample_epochs): model.get_weights()})
    )

    # predict and update model weights
    update_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch,
        logs: model.set_weights(naive_prediction(sample_weights, sample_epochs, predict_epoch*pow(pr, ((epoch + 1)/sample_epochs)) if predict_epoch*pow(pr, ((epoch + 1)/sample_epochs)) > sample_epochs + 1 else sample_epochs + 1)) if (
            epoch + 1) % sample_epochs == 0 else None
    )

    retraindir = out_prefix + \
        f'/retrain_{original_epochs}o_{sample_epochs}s_{predict_epoch}p_{pr}pr_{qubit}q_{enc}e_{pqc}pqc'
    os.makedirs(retraindir, exist_ok=True)

    history_callback = model.fit(
        x_train, y_train, epochs=(original_epochs - sample_epochs), shuffle=True, batch_size=32, validation_data=(x_val, y_val), use_multiprocessing=False, callbacks=[weight_callback, update_callback])

    retr_loss_history = history_callback.history['loss']
    retr_acc_history = history_callback.history['accuracy']
    reval_loss_history = history_callback.history['val_loss']
    reval_acc_history = history_callback.history['val_accuracy']
    np.savetxt(retraindir + '/tr_loss_history.csv',
               np.array(retr_loss_history), delimiter=",")
    np.savetxt(retraindir + '/tr_acc_history.csv',
               np.array(retr_acc_history), delimiter=",")
    np.savetxt(retraindir + '/val_loss_history.csv',
               np.array(reval_loss_history), delimiter=",")
    np.savetxt(retraindir + '/val_acc_history.csv',
               np.array(reval_acc_history), delimiter=",")

    tr_loss.append(np.array(retr_loss_history))
    tr_acc.append(np.array(retr_acc_history))
    val_loss.append(np.array(reval_loss_history))
    val_acc.append(np.array(reval_acc_history))

    var_list.append(predict_epoch - sample_epochs + 1)
    predict_epoch += 2

# Training accuracy
x1 = [i+1 for i in range(original_epochs)]
y1 = np.array(tr_acc_history)
plt.plot(x1, y1, label="Original")
for i in range(len(tr_acc)):
    y2 = np.concatenate((y1[:sample_epochs], tr_acc[i]), axis=None)
    plt.plot(x1, y2, label=f"Prediction interval: {var_list[i]}")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy')
plt.legend()
plt.savefig(traindir + '/train_acc.png')

# Training loss
plt.clf()
x1 = [i+1 for i in range(original_epochs)]
y1 = np.array(tr_loss_history)
plt.plot(x1, y1, label="Original")
for i in range(len(tr_loss)):
    y2 = np.concatenate((y1[:sample_epochs], tr_loss[i]), axis=None)
    plt.plot(x1, y2, label=f"Prediction interval: {var_list[i]}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss')
plt.legend()
plt.savefig(traindir + '/train_loss.png')

# Validation accuracy
plt.clf()
x1 = [i+1 for i in range(original_epochs)]
y1 = np.array(val_acc_history)
plt.plot(x1, y1, label="Original")
for i in range(len(val_acc)):
    y2 = np.concatenate((y1[:sample_epochs], val_acc[i]), axis=None)
    plt.plot(x1, y2, label=f"Prediction interval: {var_list[i]}")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation accuracy')
plt.legend()
plt.savefig(traindir + '/val_acc.png')

# Validation loss
plt.clf()
x1 = [i+1 for i in range(original_epochs)]
y1 = np.array(val_loss_history)
plt.plot(x1, y1, label="Original")
for i in range(len(val_loss)):
    y2 = np.concatenate((y1[:sample_epochs], val_loss[i]), axis=None)
    plt.plot(x1, y2, label=f"Prediction interval: {var_list[i]}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation loss')
plt.legend()
plt.savefig(traindir + '/val_loss.png')
