from torch.utils.data import DataLoader


def compute_accuracy(model, data, mini_batch_size=100):
    """
        Compute the number of errors the model infers on the data set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        data: Dataset
        mini_batch_size : int
            The size of the batch processing size

        Returns
        -------
        int
            The number of errors the model infers
    """
    loader = DataLoader(data, mini_batch_size)
    nb_errors = 0

    # Processing the data set by mini batches
    for data_input, target, _ in loader:
        output = model(data_input)
        if isinstance(output, tuple):
            output = next(iter(output))
        pred = output >= 0
        nb_errors += (pred != target).sum().item()

    return 1 - (nb_errors / len(data))


def compute_accuracy_classification(model, data, mini_batch_size=100):
    """
        Compute the number of errors the model infers on the data set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        data: Dataset
        mini_batch_size : int
            The size of the batch processing size

        Returns
        -------
        int
            The number of errors the model infers
    """
    loader = DataLoader(data, mini_batch_size)
    nb_errors = 0

    # Processing the data set by mini batches
    for data_input, target, _ in loader:
        output = model(data_input).argmax(dim=-1).reshape(-1, 2)
        pred = output[:, 0] <= output[:, 1]
        nb_errors += (pred != target).sum().item()

    return 1 - (nb_errors / len(data))
