import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from metrics import compute_accuracy, compute_accuracy_classification
from utils import EarlyStopping


# Training function giving directly the desired prediction,
# Used for networks : TwoChannelsModel, TwoBranchesModel, WeightSharingBranchesModel, BranchesToVecModel, WeightSharingBranchesToVecModel

def train_pred(model, train, val, mini_batch_size=100, lr=3e-4, nb_epochs=100, patience=20, **kwargs):
    """
        Train the PyTorch model on the training set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        train : TensorDataset
            Dataset containing inputs, targets, classes for training (train_inner)
        val : TensorDataset
            Dataset containing inputs, targets, classes for validation
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

    # Defining DataLoaders for better mini-batches handling
    # Shuffling makes batches differ between epochs and results in more robust training
    train_loader = DataLoader(train, mini_batch_size, shuffle=True)

    # Learning loop
    for e in range(nb_epochs):
        model.train()
        # Train the input dataset by dividing it into mini_batch_size small datasets
        for train_input, train_target, _ in train_loader:
            output = model(train_input)
            loss = criterion(output, train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss)
        model.eval()
        val_accs.append(compute_accuracy(model, val, mini_batch_size))

        # If the validation accuracy has not improved enough in the last patience epochs
        # then stop training
        if early_stopping(val_accs[-1]):
            break

    return model, train_losses, val_accs


# Training function giving the desired prediction and the labels of the two digits
# Used for network: WeightSharingAuxiliaryModel 

def train_pred_labels(model, train, val, auxiliary_weight=1., mini_batch_size=100,
                          lr=3e-4, nb_epochs=100, patience=20, **kwargs):
    """
        Train the PyTorch model on the training set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        train : TensorDataset
            Dataset containing inputs, targets, classes for training (train_inner)
        val : TensorDataset
            Dataset containing inputs, targets, classes for validation
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

    # Defining the criteria to calculate losses
    criterion = nn.BCEWithLogitsLoss()         # for Binary Classification
    criterion_digit = nn.CrossEntropyLoss()    # for MultiClass Classification

    # Defining the early stopping criterion
    early_stopping = EarlyStopping(patience)

    # Defining DataLoaders for better mini-batches handling
    # Shuffling makes batches differ between epochs and results in more robust training
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

        # If the validation accuracy has not improved enough in the last patience epochs
        # then stop training
        if early_stopping(val_accs[-1]):
            break

    return model, train_losses, val_accs


# Training function giving the labels of the two digits
# Used for network: DirectClassificationModel

def train_labels(model, train, val, mini_batch_size=100,
                     lr=3e-4, nb_epochs=100, patience=20, **kwargs):
    """
            Train the PyTorch model on the training set.

            Parameters
            ----------
            model : PyTorch NN object
                PyTorch neural network model
            train : TensorDataset
                Dataset containing inputs, targets, classes for training (train_inner)
            val : TensorDataset
                Dataset containing inputs, targets, classes for validation
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

    # Defining DataLoaders for better mini-batches handling
    # Shuffling makes batches differ between epochs and results in more robust training
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

        # If the validation accuracy has not improved enough in the last patience epochs
        # then stop training
        if early_stopping(val_accs[-1]):
            break

    return model, train_losses, val_accs
