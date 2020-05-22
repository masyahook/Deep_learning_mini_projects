import random
import torch
import dl


def main(n_epochs=200, batch_size=100, seed=42):
    """
        The main function that trains the implemented model and output train and test error.

        Parameters
        ----------
        n_epochs : int
            Number of epochs for training
        batch_size: int
            Batch size for each training epoch
        seed : int
            Random seed
    """
    
    # Setting the seed for random generator
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Turning off the autograd
    torch.set_grad_enabled(False)

    # Generating train and test data
    X_train, y_train = dl.generate_data(1000)
    X_test, y_test = dl.generate_data(1000)

    # Creating the model as a container of layers
    model = dl.Sequential([
        dl.Linear(2, 25),
        dl.ReLU(),
        dl.Linear(25, 25),
        dl.ReLU(),
        dl.Dropout(0.3),
        dl.Linear(25, 25),
        dl.Dropout(0.5),
        dl.Tanh(),
        dl.Linear(25, 1),
        dl.Sigmoid()
    ])
    
    # Defining the MSE loss
    loss = dl.LossBCE()
    
    # Defining the SGD optimizer
    opt = dl.SGD(model.param(), lr=0.1)

    # Model training for n_epochs
    for i in range(n_epochs):
        
        # Setting the model to train mode
        model.train()
        
        # Setting initial loss to zero for each epoch
        train_loss = 0.
        
        # Training model in batches of batch_size 
        for X, y in dl.batchify((X_train, y_train), batch_size=batch_size):
            
            # Setting the gradient to zero
            opt.zero_grad()
            
            # Calculating the forward pass (output of the model)
            pred = model.forward(X)
            
            # Calculating the loss of the prediction
            train_loss += loss.forward(pred, y.reshape(-1, 1)).item()
            
            # Calculating the backward pass (gradients) using backpropagation rule
            model.backward(loss.backward())
            
            # Updating the parameter values
            opt.step()
            
        # Printing the MSE loss at each epoch    
        print(f"Epoch {i}: train MSE = {train_loss}")

    # Adding empty line between results for better representation
    print()

    # Setting the model to evaluation mode
    model.eval()
    
    # Setting initial train accuracy to zero
    train_acc = 0.
    
    # Calculating train accuracy in batches
    for X, y in dl.batchify((X_train, y_train), batch_size=batch_size, shuffle=False):
        
        # Calculating the forward pass
        pred = model.forward(X)
        
        # Summing all correct train classifications
        train_acc += ((pred.reshape(-1) > 0.5) == y).sum()
        
    # Calculating train accuracy     
    train_acc /= len(y_train)

    # Setting initial test accuracy to zero
    test_acc = 0.
    
    # Calculating test accuracy in batches
    for X, y in dl.batchify((X_test, y_test), batch_size=batch_size, shuffle=False):
        
        # Calculating the forward pass
        pred = model.forward(X)
        
        # Summing all correct test classification
        test_acc += ((pred.reshape(-1) > 0.5) == y).sum()
        
    # Calculating test accuracy
    test_acc /= len(y_test)

    # Printing the results
    print(f"Final train error = {1 - train_acc}")
    print(f"Final test error = {1 - test_acc}")

# Running the main function
if __name__ == "__main__":
    main()
