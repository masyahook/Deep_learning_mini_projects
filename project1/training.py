import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from metrics import compute_accuracy, compute_accuracy_classification
from utils import EarlyStopping


def train_model(model, train, val, mini_batch_size=100, lr=1e-3, nb_epochs=20, patience=3, **kwargs):
    """
        Train the PyTorch model on the training set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        train : train dataset
        val : validation dataset
        mini_batch_size : int
            The size of the batch processing size
        lr : float
            Learning rate for the model training
        nb_epochs : int
            The number of epochs used to train the model
        patience : int
            number of epochs without val improvement for early stopping (None to disable)

        Returns
        -------

        (NN object, train loss history, val accuracy history)
    """
    train_losses = []
    val_accs = []

    # Defining the optimizer for GD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Defining the criterion to calculate loss
    criterion = nn.BCEWithLogitsLoss()

    # Defining the early stopping criterion
    early_stopping = EarlyStopping(patience)

    # Defining dataloaders
    train_loader = DataLoader(train, mini_batch_size, shuffle=True)

    # Learning loop
    for e in range(nb_epochs):
        # Train the input dataset by dividing it into mini_batch_size small datasets
        for train_input, train_target, _ in train_loader:
            output = model(train_input)
            loss = criterion(output, train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss)
        val_accs.append(compute_accuracy(model, val, mini_batch_size))

        if early_stopping(val_accs[-1]):
            break

    return model, train_losses, val_accs


def train_model_auxiliary(model, train, val, auxiliary_weight=1., mini_batch_size=100,
                          lr=1e-3, nb_epochs=20, patience=3, **kwargs):
    """
        Train the PyTorch model on the training set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        train : train dataset
        val : validation dataset
        auxiliary_weight: float
            Weight of auxiliary loss
        mini_batch_size : int
            The size of the batch processing size
        lr : float
            Learning rate for the model training
        nb_epochs : int
            The number of epochs used to train the model
        patience : int
            number of epochs without val improvement for early stopping (None to disable)

        Returns
        -------

        (NN object, train loss history, val accuracy history)
    """
    train_losses = []
    val_accs = []

    # Defining the optimizer for GD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Defining the criterion to calculate loss
    criterion = nn.BCEWithLogitsLoss()
    criterion_digit = nn.CrossEntropyLoss()

    # Defining the early stopping criterion
    early_stopping = EarlyStopping(patience)

    # Defining dataloaders
    train_loader = DataLoader(train, mini_batch_size, shuffle=True)

    # Learning loop
    for e in range(nb_epochs):
        # Train the input dataset by dividing it into mini_batch_size small datasets
        for train_input, train_target, train_class in train_loader:
            output, output_first_digit, output_second_digit = model(train_input)
            loss_comparison = criterion(output, train_target)
            loss_digits = criterion_digit(output_first_digit, train_class[:, 0]) + \
                          criterion_digit(output_second_digit, train_class[:, 1])
            loss = loss_comparison + auxiliary_weight * loss_digits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        val_accs.append(compute_accuracy(model, val, mini_batch_size))

        if early_stopping(val_accs[-1]):
            break

    return model, train_losses, val_accs


def train_model_direct_classification(model, train, val, mini_batch_size=100,
                                      lr=1e-3, nb_epochs=20, patience=3, **kwargs):
    """
            Train the PyTorch model on the training set.

            Parameters
            ----------
            model : PyTorch NN object
                PyTorch neural network model
            train : train dataset
            val : validation dataset
            mini_batch_size : int
                The size of the batch processing size
            lr : float
                Learning rate for the model training
            nb_epochs : int
                The number of epochs used to train the model
            patience : int
                number of epochs without val improvement for early stopping (None to disable)

            Returns
            -------

            (NN object, train loss history, val accuracy history)
    """
    train_losses = []
    val_accs = []

    # Defining the optimizer for GD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Defining the criterion to calculate loss
    criterion = nn.CrossEntropyLoss()

    # Defining the early stopping criterion
    early_stopping = EarlyStopping(patience)

    # Defining dataloaders
    train_loader = DataLoader(train, mini_batch_size, shuffle=False)

    # Learning loop
    for e in range(nb_epochs):
        # Train the input dataset by dividing it into mini_batch_size small datasets
        for train_input, train_target, train_class in train_loader:
            output = model(train_input.reshape(-1, 1, *train_input.shape[2:]))
            loss = criterion(output, train_class.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_accs.append(compute_accuracy_classification(model, val, mini_batch_size))

        if early_stopping(val_accs[-1]):
            break

    return model, train_losses, val_accs
