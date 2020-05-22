import argparse
import torch
from torch.utils.data import TensorDataset
from models import *
from training import *
from metrics import *
from dlc_practical_prologue import generate_pair_sets
import pickle
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DirectClassificationModel", type=str)
    parser.add_argument("--nb_hidden", default=128, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)             # Learning rate
    parser.add_argument("--patience", default=20, type=int)             # A parameter for early stopping, see utils.py  
    parser.add_argument("--auxiliary_weight", default=1., type=float)
    parser.add_argument("--mini_batch_size", default=100, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)           # Dropout parameter, 0.0 means no dropout
    parser.add_argument("--nb_epochs", default=100, type=int)
    parser.add_argument("--nb_rounds", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    return args


def run(model_fn, train_fn, metric_fn, train, test, args):
    
    # Split train TensorDataset in proper train and validation TensorDatasets, respectively
    # The dimensions were chosen to have still enough data to perform robust training
    train_inner, val = torch.utils.data.random_split(train, [800, 200])
    
    # Create the model
    model = model_fn(args.nb_hidden, args.dropout)

    # Train it 
    model, train_losses, val_accs = train_fn(model=model, train=train_inner, val=val,
                                             mini_batch_size=args.mini_batch_size, nb_epochs=args.nb_epochs,
                                             lr=args.lr, patience=args.patience, auxiliary_weight=args.auxiliary_weight)

    # Get statistics
    train_accuracy = metric_fn(model, train)
    test_accuracy = metric_fn(model, test)
    
    return model, train_accuracy, test_accuracy, val_accs


def main(args):
    # Associate to every model the corresponding train and metric function
    if args.model == "DirectClassificationModel":
        model_fn = DirectClassificationModel
        train_fn = train_labels
        metric_fn = compute_accuracy_classification
    elif args.model == "WeightSharingAuxiliaryModel":
        model_fn = WeightSharingAuxiliaryModel
        train_fn = train_pred_labels
        metric_fn = compute_accuracy
    else:
        model_fn = eval(args.model)
        train_fn = train_pred
        metric_fn = compute_accuracy

    # Store train and test accuracies over nb_rounds indipendent rounds
    
    train_accs = []
    test_accs = []
    val_accs_round = [] #MODIFIED
    for i in range(args.nb_rounds):
        
        # Get the data
        train_inputs, train_targets, train_classes, test_inputs, test_targets, test_classes = generate_pair_sets(1000)

        # Normalize the training sets - to have similar data distribution for every pixel
        mu, std = train_inputs.mean(), train_inputs.std()
        train_inputs.sub_(mu).div_(std)

        # Normalize the test sets - - to have similar data distribution for every pixel
        test_inputs.sub_(mu).div_(std)

        train_targets, test_targets = train_targets.float(), test_targets.float()
        
        # TensorDataset (later coupled with DataLoader) used for better mini-batches handling 
        test = TensorDataset(test_inputs, test_targets, test_classes)
        train = TensorDataset(train_inputs, train_targets, train_classes)
        _, train_acc, test_acc, val_accs = run(model_fn, train_fn, metric_fn, train, test, args) #modified
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        val_accs_round.append(val_accs)

        print(f"Round {i + 1}: train accuracy = {train_acc}, test accuracy = {test_acc}")

    with open(f"logs/{args.model + ('Dropout' + str(args.dropout).replace('.', '_') if args.dropout > 0 else '')}.pt", "wb") as f:
        pickle.dump(val_accs_round, f)

    return train_accs, test_accs


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    os.system("mkdir -p logs")

    train_accs, test_accs = main(args)
    
    # Convert lists to tensors and compute statistics
    train_acc_mean = torch.mean(torch.tensor(train_accs))
    train_acc_std = torch.std(torch.tensor(train_accs))
    test_acc_mean = torch.mean(torch.tensor(test_accs))
    test_acc_std = torch.std(torch.tensor(test_accs))
    
    print()
    print(f"Estimated using {args.nb_rounds} independent rounds:")
    print(f"Train accuracy: mean={train_acc_mean}, std={train_acc_std}")
    print(f"Test accuracy: mean={test_acc_mean}, std={test_acc_std}")

