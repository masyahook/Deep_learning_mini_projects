import random
import torch
import dl


def main(n_epochs=200, batch_size=100, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    X_train, y_train = dl.generate_data(1000)
    X_test, y_test = dl.generate_data(1000)

    model = dl.Sequential([
        dl.Linear(2, 25),
        dl.ReLU(),
        dl.Linear(25, 25),
        dl.ReLU(),
        dl.Linear(25, 25),
        dl.ReLU(),
        dl.Linear(25, 1)
    ])
    loss = dl.LossMSE()
    opt = dl.SGD(model.param(), lr=0.1)

    for i in range(n_epochs):
        model.train()
        train_loss = 0.
        for X, y in dl.batchify((X_train, y_train), batch_size=batch_size):
            opt.zero_grad()
            pred = model.forward(X)
            train_loss += loss.forward(pred, y.reshape(-1, 1)).item()
            model.backward(loss.backward())
            opt.step()
        print(f"Epoch {i}: train MSE = {train_loss}")

    print()

    model.eval()
    train_acc = 0.
    for X, y in dl.batchify((X_train, y_train), batch_size=batch_size, shuffle=False):
        pred = model.forward(X)
        train_acc += ((pred.reshape(-1) > 0.5) == y).sum()
    train_acc /= len(y_train)

    test_acc = 0.
    for X, y in dl.batchify((X_test, y_test), batch_size=batch_size, shuffle=False):
        pred = model.forward(X)
        test_acc += ((pred.reshape(-1) > 0.5) == y).sum()
    test_acc /= len(y_test)

    print(f"Final train error = {1 - train_acc}")
    print(f"Final test error = {1 - test_acc}")


if __name__ == "__main__":
    main()
