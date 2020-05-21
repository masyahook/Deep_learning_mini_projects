from torch.utils.data import DataLoader

# Used for all Models but DirectClassificationModel
def compute_accuracy(model, data, mini_batch_size=100):
    """
        Compute the number of errors the model infers on the data set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        data : TensorDataset
            Dataset containing inputs, targets, classes for the data (either train validation or test)
        mini_batch_size : int
            The size of the batch processing size

        Returns
        -------
        float
            The accuracy of the model
    """
    loader = DataLoader(data, mini_batch_size)
    nb_errors = 0

    # Processing the data set by mini batches
    for data_input, target, _ in loader:
        output = model(data_input)
        if isinstance(output, tuple):         # if the model returns more than one Tensor (like in WeightSharingAuxiliaryModel)
            output = next(iter(output))       # then consider only the first (e.g. the prediction value)
        pred = output >= 0                    # Boolean True/False equivalent to 1/0 in PyTorch
        nb_errors += (pred != target).sum().item()   # There is an error any time the predicted result is different from the target

    return 1 - (nb_errors / len(data))


# Used for DirectClassificationModel only
def compute_accuracy_classification(model, data, mini_batch_size=100):
    """
        Compute the number of errors the model infers on the data set.

        Parameters
        ----------
        model : PyTorch NN object
            PyTorch neural network model
        data : TensorDataset
            Dataset containing inputs, targets, classes for the data (either train validation or test)
        mini_batch_size : int
            The size of the batch processing size

        Returns
        -------
        float
            The accuracy of the model
    """
    loader = DataLoader(data, mini_batch_size)
    nb_errors = 0

    # Processing the data set by mini batches
    for data_input, target, _ in loader:
        # The model results has shape [2*mini_batch_size, 10]
        # Going row by row, we take the position of the maximum value
        # and we reshape so as to have two columns corresponding to the two images
        output = model(data_input).argmax(dim=-1).reshape(-1, 2)    
        # Where the first column (image) is smaller than the second ? 
        # Boolean True/False equivalent to 1/0 in PyTorch
        pred = output[:, 0] <= output[:, 1]   
        # There is an error any time the predicted result is different from the target
        nb_errors += (pred != target).sum().item()

    return 1 - (nb_errors / len(data))
