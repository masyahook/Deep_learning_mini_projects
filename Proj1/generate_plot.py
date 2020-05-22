import argparse
import pickle
import torch
import matplotlib.pyplot as plt


def trim_list(a):
    length = max(map(len, a))

    return [x + [x[-1]] * (length - len(x)) for x in a]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=["logs/TwoChannelsModel",
                                                       "logs/TwoBranchesModel",
                                                       "logs/WeightSharingBranchesModel",
                                                       "logs/BranchesToVecModel",
                                                       "logs/WeightSharingBranchesToVecModel",
                                                       "logs/WeightSharingAuxiliaryModel",
                                                       "logs/DirectClassificationModel"])
    parser.add_argument("--dropout", default=0.0, type=float)
    conf = parser.parse_args()
    plots = dict()

    for name in conf.files:
        if conf.dropout > 0:
            filename = name + f"Dropout{conf.dropout}".replace(".", "_") + ".pt"
        else:
            filename = name + ".pt"
        with open(filename, "rb") as f:
            plots[name] = torch.tensor(trim_list(pickle.load(f))).mean(dim=0).tolist()

    plt.figure(figsize=(10, 6))
    plt.grid(":")

    plt.title(f"Validation accuracies history{', dropout ' + str(conf.dropout) if conf.dropout > 0 else ''}")
    for name in conf.files:
        model = name.split("/")[-1]
        plt.plot(range(len(plots[name])), plots[name], label=model)
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.savefig(f"plot_dropout_{conf.dropout}".replace('.', '_') if conf.dropout > 0 else "plot.png", bbox_inches="tight", dpi=700)
