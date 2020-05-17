import argparse
import torch
from torch.utils.data import TensorDataset
from models import *
from training import *
from metrics import *
from dlc_practical_prologue import generate_pair_sets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DirectClassification", type=str)
    parser.add_argument("--nb_hidden", default=128, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--auxiliary_weight", default=1., type=float)
    parser.add_argument("--mini_batch_size", default=100, type=int)
    parser.add_argument("--nb_epochs", default=100, type=int)
    parser.add_argument("--nb_rounds", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    return args


def run(model_fn, train_fn, metric_fn, train, test, args):
    train_inner, val = torch.utils.data.random_split(train, [800, 200])
    model = model_fn(args.nb_hidden)

    model, train_losses, val_accs = train_fn(model=model, train=train_inner, val=val,
                                             mini_batch_size=args.mini_batch_size, nb_epochs=args.nb_epochs,
                                             lr=args.lr, patience=args.patience, auxiliary_weight=args.auxiliary_weight)

    train_accuracy = metric_fn(model, train)
    test_accuracy = metric_fn(model, test)

    return model, train_accuracy, test_accuracy


def main(args):
    if args.model == "DirectClassification":
        model_fn = DirectClassification
        train_fn = train_model_direct_classification
        metric_fn = compute_accuracy_classification
    elif args.model == "TwoBranchSecondWeightSharingAuxiliaryLoss":
        model_fn = TwoBranchSecondWeightSharingAuxiliaryLoss
        train_fn = train_model_auxiliary
        metric_fn = compute_accuracy
    else:
        model_fn = eval(args.model)
        train_fn = train_model
        metric_fn = compute_accuracy

    train_accs = []
    test_accs = []
    for _ in range(args.nb_rounds):
        train_inputs, train_targets, train_classes, test_inputs, test_targets, test_classes = generate_pair_sets(1000)

        # Normalize the training sets
        mu, std = train_inputs.mean(), train_inputs.std()
        train_inputs.sub_(mu).div_(std)

        # Normalize the test sets
        test_inputs.sub_(mu).div_(std)

        train_targets, test_targets = train_targets.float(), test_targets.float()
        test = TensorDataset(test_inputs, test_targets, test_classes)
        train = TensorDataset(train_inputs, train_targets, train_classes)
        _, train_acc, test_acc = run(model_fn, train_fn, metric_fn, train, test, args)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return train_accs, test_accs


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    train_accs, test_accs = main(args)
    train_acc_mean = torch.mean(torch.tensor(train_accs))
    train_acc_std = torch.std(torch.tensor(train_accs))
    test_acc_mean = torch.mean(torch.tensor(test_accs))
    test_acc_std = torch.std(torch.tensor(test_accs))

    print(f"Estimated using {args.nb_rounds} independent rounds:")
    print(f"Train accuracy: mean={train_acc_mean}, std={train_acc_std}")
    print(f"Test accuracy: mean={test_acc_mean}, std={test_acc_std}")

