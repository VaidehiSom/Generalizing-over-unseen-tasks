import gzip
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from util import \
    available_actions, \
    data_transform, \
    DATA_DIR, \
    DATA_FILE, \
    MODEL_FILE

restore = False  # restore from file if exists
BATCH_SIZE = 0 # TODO: decide batch size
EPOCHS = 0 # TODO: decide no of epochs
TRAIN_VAL_SPLIT = 0.85  # train/val ratio


def read_data():
    # TODO: Balance dataset- drop some actions that are very common, multiply rare events

    """Read the data generated"""
    with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
        data = pickle.load(f)

    random.shuffle(data) # we'll shuffle data

    # mapping data to numpy arrays
    states, actions = map(np.array, zip(*data))

    # reverse one-hot, actions to classes
    act_classes = np.full((len(actions)), -1, dtype=np.int)
    for i, a in enumerate(available_actions): # i is row number, a is available action row
        act_classes[np.all(actions == a, axis=1)] = i

    print("Total transitions: " + str(len(act_classes))) # to check if everything correct

    return states, act_classes


def create_datasets():
    """Create training and validation datasets"""

    class TensorDatasetTransforms(torch.utils.data.TensorDataset):
        """Helper class to allow transformations (by default TensorDataset doesn't support them)"""

        def __init__(self, x, y):
            super().__init__(x, y)

        def __getitem__(self, index):
            tensor = data_transform(self.tensors[0][index])
            return (tensor,) + tuple(t[index] for t in self.tensors[1:])

    x, y = read_data()
    x = np.moveaxis(x, 3, 1)  # channel first (torch requirement)

    # train dataset
    x_train = x[:int(len(x) * TRAIN_VAL_SPLIT)]
    y_train = y[:int(len(y) * TRAIN_VAL_SPLIT)]

    train_set = TensorDatasetTransforms(
        torch.tensor(x_train),
        torch.tensor(y_train))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)

    # test dataset
    x_val, y_val = x[int(len(x_train)):], y[int(len(y_train)):]

    val_set = TensorDatasetTransforms(
        torch.tensor(x_val),
        torch.tensor(y_val))

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=2)

    return train_loader, val_loader


def build_network():
    """Build the torch network"""

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)

    # TODO: Decide on model layers, hyperparams, etc
    model = torch.nn.Sequential(
        torch.nn.Conv2d(),
        torch.nn.BatchNorm2d(),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.5),
        torch.nn.Conv2d(),
        torch.nn.BatchNorm2d(),
        torch.nn.ELU(),
        torch.nn.Dropout2d(),
        torch.nn.Conv2d(),
        torch.nn.ELU(),
        Flatten(),
        torch.nn.BatchNorm1d(),
        torch.nn.Dropout(),
        torch.nn.Linear(),
        torch.nn.ELU(),
        torch.nn.BatchNorm1d(),
        torch.nn.Dropout(),
        torch.nn.Linear(),
    )

    return model


def train(model, device):
    """Training main method"""

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    train_loader, val_order = create_datasets()  # read datasets

    # train
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        train_epoch(model,
                    device,
                    loss_function,
                    optimizer,
                    train_loader)

        test(model, device, loss_function, val_order)

        # save model
        model_path = os.path.join(DATA_DIR, MODEL_FILE)
        torch.save(model.state_dict(), model_path)


def train_epoch(model, device, loss_function, optimizer, data_loader):
    """Train for a single epoch"""

    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test(model, device, loss_function, data_loader):
    """Test over the whole dataset"""

    # set model in evaluation mode
    model.eval()  
    
    current_loss = 0.0
    current_acc = 0

    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}' .format(total_loss, total_acc))