import torch
from torch import nn
from torch.nn import functional as F


class TwoChannel(nn.Module):
    def __init__(self, nb_hidden=128):
        super(TwoChannel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(-1)

        return x


class TwoBranch(nn.Module):
    def __init__(self, nb_hidden=128):
        super(TwoBranch, self).__init__()

        # Convolutional layers on the first branch
        self.cnn_first = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers on the first branch
        self.fc_first = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, 10),
            nn.ReLU(),
            nn.Linear(10, 1)

        )

        # Convolutional layers on the second branch
        self.cnn_second = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers on the second branch
        self.fc_second = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, 10),
            nn.ReLU(),
            nn.Linear(10, 1)

        )

    def forward_first(self, x):
        x = self.cnn_first(x)
        x = x.view(x.size(0), -1)
        x = self.fc_first(x)
        return x

    def forward_second(self, x):
        x = self.cnn_second(x)
        x = x.view(x.size(0), -1)
        x = self.fc_second(x)
        return x

    def forward(self, x):
        output_1 = self.forward_first(x[:, 0, None])
        output_2 = self.forward_second(x[:, 1, None])
        output = (output_2 - output_1).view(-1)
        return output


class TwoBranchWeightSharing(nn.Module):
    def __init__(self, nb_hidden=128):
        super(TwoBranchWeightSharing, self).__init__()

        # Convolutional layers in each branch
        self.cnn_single = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers in each branch
        self.fc_single = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, 10),
            nn.ReLU(),
            nn.Linear(10, 1)

        )

    def forward_single(self, x):
        x = self.cnn_single(x)
        x = x.view(x.size(0), -1)
        x = self.fc_single(x)
        return x

    def forward(self, x):
        output_1 = self.forward_single(x[:, 0, None])
        output_2 = self.forward_single(x[:, 1, None])
        output = (output_2 - output_1).view(-1)
        return output


class TwoBranchSecond(nn.Module):
    def __init__(self, nb_hidden=128):
        super(TwoBranchSecond, self).__init__()

        # Convolutional layers on the first branch
        self.cnn_first = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers on the first branch
        self.fc_first = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, 10),
            nn.ReLU(),

        )

        # Convolutional layers on the second branch
        self.cnn_second = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers on the second branch
        self.fc_second = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, 10),
            nn.ReLU(),

        )

        self.fc_prefinal = nn.Linear(20, 10)
        self.fc_final = nn.Linear(10, 1)

    def forward_first(self, x):
        x = self.cnn_first(x)
        x = x.view(x.size(0), -1)
        x = self.fc_first(x)
        return x

    def forward_second(self, x):
        x = self.cnn_second(x)
        x = x.view(x.size(0), -1)
        x = self.fc_second(x)
        return x

    def forward(self, x):
        output_1 = self.forward_first(x[:, 0, None])
        output_2 = self.forward_second(x[:, 1, None])
        output = torch.cat((output_1, output_2), 1)
        output = F.relu(self.fc_prefinal(output))
        output = self.fc_final(output).view(-1)
        return output


class TwoBranchSecondWeightSharing(nn.Module):
    def __init__(self, nb_hidden=128):
        super(TwoBranchSecondWeightSharing, self).__init__()

        # Convolutional layers in each branch
        self.cnn_single = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers in each branch
        self.fc_single = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),
            nn.Linear(nb_hidden, 10),
            nn.ReLU(),

        )

        self.fc_prefinal = nn.Linear(20, 10)
        self.fc_final = nn.Linear(10, 1)

    def forward_single(self, x):
        x = self.cnn_single(x)
        x = x.view(x.size(0), -1)
        x = self.fc_single(x)
        return x

    def forward(self, x):
        output_1 = self.forward_single(x[:, 0, None])
        output_2 = self.forward_single(x[:, 1, None])
        output = torch.cat((output_1, output_2), 1)
        output = F.relu(self.fc_prefinal(output))
        output = self.fc_final(output).view(-1)
        return output


class TwoBranchSecondWeightSharingAuxiliaryLoss(nn.Module):
    def __init__(self, nb_hidden=128):
        super(TwoBranchSecondWeightSharingAuxiliaryLoss, self).__init__()

        # Convolutional layers in each branch
        self.cnn_single = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()

        )

        # Fully-connected layers in each branch
        self.fc_single = nn.Sequential(

            nn.Linear(256, nb_hidden),
            nn.ReLU(),

        )

        self.fc_prefinal = nn.Linear(2 * nb_hidden, 100)
        self.fc_comparison = nn.Linear(100, 1)
        self.fc_first_digit = nn.Linear(100, 10)
        self.fc_second_digit = nn.Linear(100, 10)

    def forward_single(self, x):
        x = self.cnn_single(x)
        x = x.view(x.size(0), -1)
        x = self.fc_single(x)
        return x

    def forward(self, x):
        output_1 = self.forward_single(x[:, 0, None])
        output_2 = self.forward_single(x[:, 1, None])
        output = torch.cat((output_1, output_2), 1)
        output = F.relu(self.fc_prefinal(output))
        output_comparison = self.fc_comparison(output).view(-1)
        output_first_digit = self.fc_first_digit(output)
        output_second_digit = self.fc_second_digit(output)

        return output_comparison, output_first_digit, output_second_digit


class DirectClassification(nn.Module):
    def __init__(self, nb_hidden=128):
        super(DirectClassification, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward_single(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).view(-1, 10)

    def forward(self, x):
        return self.forward_single(x.reshape(-1, 1, *x.shape[2:]))
